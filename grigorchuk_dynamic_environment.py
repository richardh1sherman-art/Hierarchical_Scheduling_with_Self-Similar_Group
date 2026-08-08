import torch
import numpy as np
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ALGEBRAIC LAYER: YOUR GRIGORCHUK COMPILER
# ==========================================
class TrueBottomUpGrigorchukCompiler:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.a_mat, self.b_mat, self.c_mat, self.d_mat = self._build_tree_layer(max_depth)
        self.dim = 2 ** max_depth

    def _build_tree_layer(self, current_depth):
        if current_depth == 1:
            a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
            b = torch.eye(2, device=device)
            c = torch.eye(2, device=device)
            d = torch.eye(2, device=device)
            return a, b, c, d

        sub_depth = current_depth - 1
        sub_dim = 2 ** sub_depth
        dim_local = 2 ** current_depth
        
        a_sub, b_sub, c_sub, d_sub = self._build_tree_layer(sub_depth)
        sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        
        a_local = torch.kron(sigma_x, torch.eye(sub_dim, device=device))
        b_local = torch.zeros(dim_local, dim_local, device=device)
        b_local[:sub_dim, :sub_dim] = a_sub
        b_local[sub_dim:, sub_dim:] = c_sub

        c_local = torch.zeros(dim_local, dim_local, device=device)
        c_local[:sub_dim, :sub_dim] = a_sub
        c_local[sub_dim:, sub_dim:] = d_sub

        d_local = torch.zeros(dim_local, dim_local, device=device)
        d_local[:sub_dim, :sub_dim] = torch.eye(sub_dim, device=device)
        d_local[sub_dim:, sub_dim:] = b_sub

        return a_local, b_local, c_local, d_local

# ==========================================
# AGENT 1: THE GRIGORCHUK LOOK-AHEAD ROUTER
# ==========================================
class DynamicGroupRouter(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.encoder = torch.nn.Linear(dim * 2, 128)
        self.scorer = torch.nn.Linear(128, 1)

    def forward(self, current_state, candidate_states):
        scores = []
        for i in range(4):
            combined = torch.cat([current_state, candidate_states[i]], dim=-1)
            x = torch.relu(self.encoder(combined))
            score = self.scorer(x)
            scores.append(score)
        return torch.softmax(torch.cat(scores, dim=-1), dim=-1)

# ==========================================================
# AGENT 2: ADVERSARIAL EXTERNAL ENVIRONMENT GENERATOR
# ==========================================================
class AdversarialRestaurantEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        # 4 active tables: 0=Idle, 1=Ordered, 2=Served Coffee, 3=Order Fulfilled
        self.table_stages = np.zeros(4, dtype=int)
        self.coffee_machine_broken = False

    def trigger_random_environmental_shock(self, step_tick):
        # Shock Type 1: A customer changes their mind, resetting fulfillment criteria
        if step_tick == 15 and random.random() < 0.70:
            target_table = random.randint(0, 3)
            self.table_stages[target_table] = 0 
            return f"⚠️ SHOCK: Table {target_table} changed their order mind!"

        # Shock Type 2: At step 25, the coffee machine breaks
        if step_tick == 25 and random.random() < 0.50:
            self.coffee_machine_broken = True
            return "⚠️ SHOCK: Main Coffee pot is bottlenecked/busy!"
            
        if step_tick == 35:
            self.coffee_machine_broken = False 
            
        return None

    def evaluate_state_action(self, prev_idx, curr_idx):
        if prev_idx == curr_idx:
            return -1.0  # Deadlock/Idle penalty
            
        table_focus = (curr_idx >> 2) & 3
        task_action  = curr_idx & 3
        current_stage = self.table_stages[table_focus]
        
        reward = 0.0
        
        if self.coffee_machine_broken and task_action == 2:
            return -3.0  # High delay penalty
            
        if task_action == 1 and current_stage == 0:
            self.table_stages[table_focus] = 1
            reward += 20.0
        elif task_action == 2 and current_stage == 1:
            self.table_stages[table_focus] = 2
            reward += 40.0
        elif task_action == 3 and current_stage == 2:
            self.table_stages[table_focus] = 3
            reward += 80.0
        else:
            reward -= 0.5 
            
        # NATIVE WORD METRIC DISTANCE FEEDBACK (Hamming tree alignment)
        # Keeps gradients active even when shocks push the index away from targets
        tree_dist = bin(curr_idx ^ 15).count('1')
        reward += (5.0 - float(tree_dist)) * 2.0
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================
# ADVERSARIAL RUNNER LOOP
# ==========================================
def run_dynamic_adversarial_experiment(iterations=250, horizon=50):
    print("====================================================================")
    print("🤖 STEP 2: LAUNCHING ADVERSARIAL TWO-AGENT CO-EVOLUTIONARY LOOP")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env_agent = AdversarialRestaurantEnvironment()
    scheduler_agent = DynamicGroupRouter(dim=32).to(device)
    optimizer = torch.optim.Adam(scheduler_agent.parameters(), lr=0.003)
    
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    score_history = []
    
    for iteration in range(iterations):
        env_agent.reset()
        
        # FIXED: Correct one-hot tensor initialization prevents float type collisions
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0  
        
        log_probs = []
        rewards = []
        
        for step in range(horizon):
            shock_alert = env_agent.trigger_random_environmental_shock(step)
            if shock_alert and iteration % 50 == 0:
                print(f"    [Iteration {iteration+1:03d} | Step {step:02d}] {shock_alert}")
                
            candidate_vectors = []
            for gen in generators:
                candidate_vectors.append(torch.matmul(gen, state_vector).unsqueeze(0))
            candidate_tensor = torch.cat(candidate_vectors, dim=0)
            
            probs = scheduler_agent(state_vector, candidate_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            next_state_vector = candidate_tensor[action.item()]
            
            prev_idx = torch.argmax(state_vector).item()
            curr_idx = torch.argmax(next_state_vector).item()
            
            step_reward = env_agent.evaluate_state_action(prev_idx, curr_idx)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            
        # Policy Gradient Optimization Backprop
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.96 * R
            discounted.insert(0, R)
        discounted = torch.tensor(discounted, device=device)
        if discounted.std() > 1e-5:
            discounted = (discounted - discounted.mean()) / (discounted.std())
            
        loss = 0
        for lp, rw in zip(log_probs, discounted):
            loss -= lp * rw
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        final_score = env_agent.get_filled_orders_count()
        score_history.append(final_score)
        
        if (iteration + 1) % 25 == 0:
            print(f"  Iteration {iteration+1:03d} -> Orders Successfully Filled Despite Shocks: {final_score}/4 Tables")
            
    plt.figure(figsize=(10, 5))
    plt.plot(score_history, color='darkorange', linewidth=2, label='Resilient Grigorchuk Net')
    plt.title('Dynamic Scheduling Adaptation Under Adversarial Environmental Shocks')
    plt.xlabel('Training Iterations (Co-Evolution Steps)')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('dynamic_adversarial_performance.png')
    print("\n📊 PLOT GENERATED: Multi-agent tracking graph saved as 'dynamic_adversarial_performance.png'")

if __name__ == "__main__":
    run_dynamic_adversarial_experiment()

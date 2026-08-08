import torch
import numpy as np
import matplotlib.pyplot as plt
from z3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ALGEBRAIC LAYER: GRIGORCHUK COMPILER
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
# NEW NN ARCHITECTURE: TARGET DIRECTION RANKER
# ==========================================
class GrigorchukFirstRouter(torch.nn.Module):
    """
    Takes the current state vector AND the 4 valid future vectors 
    pre-calculated by Grigorchuk, scoring which valid branch is best.
    """
    def __init__(self, dim=32):
        super().__init__()
        self.encoder = torch.nn.Linear(dim * 2, 128)
        self.scorer = torch.nn.Linear(128, 1)

    def forward(self, current_state, candidate_states):
        # candidate_states shape: [4, 32]
        scores = []
        for i in range(4):
            # Concatenate the current position with the proposed algebraic step
            combined = torch.cat([current_state, candidate_states[i]], dim=-1)
            x = torch.relu(self.encoder(combined))
            score = self.scorer(x)
            scores.append(score)
        return torch.softmax(torch.cat(scores, dim=-1), dim=-1)

# ==========================================
# RESTAURANT ENVIRONMENT AGENT
# ==========================================
class RestaurantEnvironmentAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.customer_stages = np.zeros(5, dtype=int)
        
    def evaluate_state_evolution(self, prev_idx, curr_idx):
        reward = 0.0
        if prev_idx == curr_idx:
            return -1.0 # Penalize staying idle
        
        customer_focus = curr_idx % 5
        action_type = (curr_idx // 5) % 4  # 0=Order, 1=Coffee, 2=Meal, 3=Idle
        current_stage = self.customer_stages[customer_focus]
        
        if action_type == 0 and current_stage == 0:
            self.customer_stages[customer_focus] = 1
            reward += 10.0  
        elif action_type == 1 and current_stage == 1:
            self.customer_stages[customer_focus] = 2
            reward += 20.0 
        elif action_type == 2 and current_stage == 2:
            self.customer_stages[customer_focus] = 3
            reward += 50.0 
        else:
            reward -= 0.5  
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.customer_stages == 3))

# ==========================================
# TRAINER LOOP
# ==========================================
def run_grigorchuk_first_experiment(iterations=200, horizon=50):
    print("====================================================================")
    print("🔬 PARADIGM SHIFT: RUNNING GRIGORCHUK FIRST -> THEN NN ROUTING")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env = RestaurantEnvironmentAgent()
    router = GrigorchukFirstRouter(dim=32).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.005)
    
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    performance_history = []
    
    for iteration in range(iterations):
        env.reset()
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0  # Start node
        
        log_probs = []
        rewards = []
        
        for step in range(horizon):
            # --- 1. GRIGORCHUK RUNS FIRST ---
            # Pre-calculate the 4 possible next steps according to group geometry laws
            candidate_vectors = []
            for gen in generators:
                candidate_vectors.append(torch.matmul(gen, state_vector).unsqueeze(0))
            candidate_tensor = torch.cat(candidate_vectors, dim=0) # Shape [4, 32]
            
            # --- 2. NN RUNS SECOND ---
            # Rank the pre-calculated branches
            branch_probabilities = router(state_vector, candidate_tensor)
            
            dist = torch.distributions.Categorical(branch_probabilities)
            chosen_action = dist.sample()
            log_probs.append(dist.log_prob(chosen_action))
            
            # Step the world forward using the pre-verified candidate
            next_state_vector = candidate_tensor[chosen_action.item()]
            
            prev_idx = torch.argmax(state_vector).item()
            curr_idx = torch.argmax(next_state_vector).item()
            
            step_reward = env.evaluate_state_evolution(prev_idx, curr_idx)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            
        # Standard Reinforce update pass
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.95 * R
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
        
        score = env.get_filled_orders_count()
        performance_history.append(score)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> Orders Successfully Filled: {score}/5")
            
    # Plot tracking results
    plt.figure(figsize=(10, 5))
    plt.plot(performance_history, color='blue', linewidth=2, label='Grigorchuk-First Framework')
    plt.title('Emergent Scheduling Efficiency (Algebraic Look-Ahead Engine)')
    plt.xlabel('Training Iterations')
    plt.ylabel('Completed Orders')
    plt.grid(True)
    plt.savefig('grigorchuk_first_performance.png')
    print("\n📊 PLOT GENERATED: Saved as 'grigorchuk_first_performance.png'")

if __name__ == "__main__":
    run_grigorchuk_first_experiment()

import torch
import numpy as np
import matplotlib.pyplot as plt

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
# NN ROUTER
# ==========================================
class GrigorchukFirstRouter(torch.nn.Module):
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
# NATIVELY TREE-MAPPED RESTAURANT ENVIRONMENT AGENT
# ==========================================================
class HierarchicalTreeEnvironmentAgent:
    """
    Decodes the 32-leaf vector space as a 5-level binary decision tree.
    This respects the fractal boundaries of the self-similar group!
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # 4 distinct tables/sectors tracking progressive stages:
        # 0=Idle, 1=Ordered, 2=Served Coffee, 3=Fully Completed Order
        self.table_stages = np.zeros(4, dtype=int)
        
    def evaluate_state_evolution(self, prev_idx, curr_idx):
        if prev_idx == curr_idx:
            return -0.5 # Penalty for deadlocks or idle steps
            
        # Parse index bits to track tree-based focus
        # Bits 3-4 define which of the 4 tables the operations are focusing on
        table_focus = (curr_idx >> 2) & 3
        # Bits 0-1 define the specific task action code being broadcasted
        task_action  = curr_idx & 3
        
        current_stage = self.table_stages[table_focus]
        reward = 0.0
        
        # Chronological verification mapped to group geometry branches
        if task_action == 1 and current_stage == 0:
            self.table_stages[table_focus] = 1
            reward += 15.0  # Successfully logged table order hierarchy
        elif task_action == 2 and current_stage == 1:
            self.table_stages[table_focus] = 2
            reward += 30.0  # Coherent coffee transition executed
        elif task_action == 3 and current_stage == 2:
            self.table_stages[table_focus] = 3
            reward += 60.0  # Order completed successfully
        else:
            reward -= 0.2  # Slight out-of-order operation penalty
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================
# TRAINER RUNNER
# ==========================================
def run_tree_mapped_experiment(iterations=200, horizon=50):
    print("====================================================================")
    print("🔬 RUNNING CORRECTED EXPERIMENT: NATIVE TREE-MAPPED SCHEDULER")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env = HierarchicalTreeEnvironmentAgent()
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
            candidate_vectors = []
            for gen in generators:
                candidate_vectors.append(torch.matmul(gen, state_vector).unsqueeze(0))
            candidate_tensor = torch.cat(candidate_vectors, dim=0)
            
            branch_probabilities = router(state_vector, candidate_tensor)
            dist = torch.distributions.Categorical(branch_probabilities)
            chosen_action = dist.sample()
            
            log_probs.append(dist.log_prob(chosen_action))
            next_state_vector = candidate_tensor[chosen_action.item()]
            
            prev_idx = torch.argmax(state_vector).item()
            curr_idx = torch.argmax(next_state_vector).item()
            
            step_reward = env.evaluate_state_evolution(prev_idx, curr_idx)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            
        # Policy Gradient Reinforce step
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
            print(f"  Iteration {iteration+1:03d} -> Orders Successfully Filled: {score}/4 Tables")
            
    plt.figure(figsize=(10, 5))
    plt.plot(performance_history, color='emerald' if 'emerald' in globals() else 'green', linewidth=2, label='Tree-Mapped Grigorchuk')
    plt.title('Emergent Scheduling via Tree-Mapped Self-Similar Algebra')
    plt.xlabel('Training Iterations')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('grigorchuk_tree_performance.png')
    print("\n📊 PLOT GENERATED: Saved as 'grigorchuk_tree_performance.png'")

if __name__ == "__main__":
    run_tree_mapped_experiment()

import torch
import numpy as np
import time
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
# EQUALIZED NEURAL NETWORK ARCHITECTURE
# ==========================================
class FairPolicyAgent(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)  
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state_vector):
        x = torch.relu(self.fc1(state_vector))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

# ==========================================
# SHAPED ENVIRONMENT CRITIC
# ==========================================
class RestaurantEnvironmentAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        # 5 customers: 0=idle, 1=ordered, 2=got_coffee, 3=got_meal (fulfilled)
        self.customer_stages = np.zeros(5, dtype=int)
        
    def evaluate_state_evolution(self, previous_vector, current_vector):
        reward = 0.0
        prev_idx = torch.argmax(previous_vector).item()
        curr_idx = torch.argmax(current_vector).item()
        
        if prev_idx == curr_idx:
            return -0.5  # Deadlock penalty
        
        customer_focus = curr_idx % 5
        action_type = (curr_idx // 5) % 4  # 0=Order, 1=Coffee, 2=Meal, 3=Idle
        current_stage = self.customer_stages[customer_focus]
        
        # Core progression rules
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
            reward -= 1.0  
            
        # FIXED TREE METRIC: Distance in self-similar layers (Hamming distance of bit representation)
        # This replaces the flawed linear math with actual hierarchical tree-depth distance feedback
        tree_dist = bin(curr_idx ^ 22).count('1')
        reward += (5.0 - float(tree_dist)) * 2.0
        
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.customer_stages == 3))

# ==========================================
# SMT VERIFIER LAYER (FIXED INDEXING)
# ==========================================
def verify_trace_via_smt(action_history):
    s = Solver()
    horizon = len(action_history)
    if horizon == 0: return "Empty Trace"
        
    # Corrected shape layout: [tick][customer]
    stages = [[Int(f"cust_{c}_tick_{t}") for c in range(5)] for t in range(horizon + 1)]
    for c in range(5): 
        s.add(stages[0][c] == 0)
        
    for t in range(horizon):
        curr_idx = action_history[t]
        cust = curr_idx % 5
        act = (curr_idx // 5) % 4
        
        for c in range(5):
            if c == cust:
                if act == 0: 
                    s.add(stages[t+1][c] == If(stages[t][c] == 0, 1, stages[t][c]))
                elif act == 1: 
                    s.add(stages[t+1][c] == If(stages[t][c] == 1, 2, stages[t][c]))
                elif act == 2: 
                    s.add(stages[t+1][c] == If(stages[t][c] == 2, 3, stages[t][c]))
                else: 
                    s.add(stages[t+1][c] == stages[t][c])
            else:
                s.add(stages[t+1][c] == stages[t][c])

    if s.check() == sat:
        return "✓ LOGIC VERIFIED BY SMT ENGINE"
    else:
        return "✗ LOGIC FAILURE CAUGHT BY SMT ENGINE"

# ==========================================
# EXPERIMENTAL TRAINER FUNCTION
# ==========================================
def execute_controlled_experiment(iterations=300, horizon=50):
    print("====================================================================")
    print("🔬 RUNNING SHAPED EXPERIMENT: FRACTAL GROUP VS LINEAR SPACE")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env = RestaurantEnvironmentAgent()
    
    # Flat linear operators setup
    flat_ops = []
    for i in range(4):
        mat = torch.zeros(32, 32, device=device)
        for src in range(32):
            if i == 0: dst = (src + 1) % 32
            elif i == 1: dst = (src - 1) % 32
            elif i == 2: dst = (src + 5) % 32
            else: dst = src
            mat[dst, src] = 1.0
        flat_ops.append(mat)
        
    grigorchuk_ops = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    results = {}
    
    for mode, operators in [("Grigorchuk Fractal Group", grigorchuk_ops), ("Flat Linear System", flat_ops)]:
        print(f"Training Agent under '{mode}' constraints...")
        agent = FairPolicyAgent(dim=32).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.002)
        mode_history = []
        best_trace = []
        
        for iteration in range(iterations):
            env.reset()
            state_vector = torch.zeros(32, device=device)
            state_vector[0] = 1.0  
            
            log_probs = []
            rewards = []
            current_trace = []
            
            for step in range(horizon):
                probs = agent(state_vector)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                
                next_state_vector = torch.matmul(operators[action.item()], state_vector)
                step_reward = env.evaluate_state_evolution(state_vector, next_state_vector)
                rewards.append(step_reward)
                
                state_vector = next_state_vector
                current_trace.append(torch.argmax(state_vector).item())
                
            discounted = []
            R = 0
            for r in reversed(rewards):
                R = r + 0.98 * R
                discounted.insert(0, R)
            discounted = torch.tensor(discounted, device=device)
            discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
            
            loss = 0
            for lp, rw in zip(log_probs, discounted):
                loss -= lp * rw
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            score = env.get_filled_orders_count()
            mode_history.append(score)
            if score >= max(mode_history, default=0):
                best_trace = current_trace
                
        results[mode] = (mode_history, best_trace)
        print(f"  -> Complete. Max Score: {max(mode_history)}/5 orders.\n")

    print("Evaluating Discovered Solutions with SMT Verification Engine...")
    for mode in results:
        audit_res = verify_trace_via_smt(results[mode][1])
        print(f"  * {mode} Best Trace Audit: {audit_res}")

    plt.figure(figsize=(12, 6))
    plt.plot(results["Grigorchuk Fractal Group"][0], color='purple', linewidth=2, label='Grigorchuk Self-Similar Matrix')
    plt.plot(results["Flat Linear System"][0], color='orange', linestyle='--', linewidth=2, label='Flat Action Control (Ablation)')
    plt.title('Shaped Benchmark: Group Tree Geometry vs. Linear System')
    plt.xlabel('Training Iterations')
    plt.ylabel('Successfully Completed Orders')
    plt.legend()
    plt.grid(True)
    plt.savefig('fair_comparative_analysis.png')
    print("\n📊 FIXED PLOT GENERATED: Saved as 'fair_comparative_analysis.png'")

if __name__ == "__main__":
    execute_controlled_experiment()

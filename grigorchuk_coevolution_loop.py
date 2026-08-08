import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from z3 import *

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
# NEURAL ROUTER WITH CROSS-ENTROPY LEARNING
# ==========================================
class GrigorchukCoevolutionRouter(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)

    def forward(self, state_vector):
        x = torch.relu(self.fc1(state_vector))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================================
# ADVERSARIAL ENVIRONMENT & SMT LOCAL PATCH CALCULATOR
# ==========================================================
class CoevolutionaryEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.table_stages = np.zeros(4, dtype=int)
        
    def inject_realtime_shock(self, step_tick):
        """Randomly alters focus states to derail the neural agent path."""
        if step_tick == 15 and random.random() < 0.60:
            target_table = random.randint(0, 3)
            self.table_stages[target_table] = 0  # Forces a rollback error
            return target_table
        return None

    def calculate_smt_remediation_patch(self, current_index, broken_table):
        """
        SYMBOLIC FALLBACK (Step 3 Logic): Uses Z3 to quickly calculate an
        algebraic word patch to repair the tree deviation caused by the environment.
        """
        opt = Optimize()
        patch_horizon = 4
        
        states = [Int(f"p_state_{t}") for t in range(patch_horizon + 1)]
        actions = [Int(f"p_act_{t}") for t in range(patch_horizon)]
        
        opt.add(states == current_index)
        
        for t in range(patch_horizon):
            opt.add(actions[t] >= 0, actions[t] <= 3)
            next_state_logic = If(actions[t] == 0, states[t] + 8,
                                 If(actions[t] == 1, states[t] + 1, states[t]))
            opt.add(states[t+1] == (next_state_logic % 32))
            
        # Target: Recover structural synchronization
        opt.add(states[patch_horizon] == (broken_table * 4 + 1))
        
        if opt.check() == sat:
            m = opt.model()
            return [m[actions[t]].as_long() for t in range(patch_horizon)], [m[states[t]].as_long() for t in range(patch_horizon)]
        
        # FIXED: Correct clean layout return syntax removing the dangling tuple comma
        return [0] * 4, [current_index] * 4

    def process_direct_index(self, curr_idx):
        table_focus = (curr_idx >> 2) & 3
        task_action  = curr_idx & 3
        current_stage = self.table_stages[table_focus]
        
        if task_action == 1 and current_stage == 0:
            self.table_stages[table_focus] = 1
        elif task_action == 2 and current_stage == 1:
            self.table_stages[table_focus] = 2
        elif task_action == 3 and current_stage == 2:
            self.table_stages[table_focus] = 3

    def get_filled_orders_count(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================
# STEP 3 CO-EVOLUTIONARY TRAINER LOOP
# ==========================================
def run_step3_coevolution_experiment(iterations=150, horizon=40):
    print("====================================================================")
    print("🔬 STEP 3: RUNNING INTERACTIVE DUAL-AGENT CO-EVOLUTIONARY LOOP")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env = CoevolutionaryEnvironment()
    router = GrigorchukCoevolutionRouter(dim=32).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    performance_curve = []
    
    for iteration in range(iterations):
        env.reset()
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0  # Safe One-Hot Initialization
        
        collected_inputs = []
        collected_targets = []
        
        for step in range(horizon):
            curr_idx = torch.argmax(state_vector).item()
            
            # Check for environmental shock intervention
            broken_table = env.inject_realtime_shock(step)
            if broken_table is not None:
                # SMT intervening helper kicks in to compute an emergency correction path
                patch_actions, patch_states = env.calculate_smt_remediation_patch(curr_idx, broken_table)
                
                # Append repair data into training caches
                for s_idx, a_idx in zip(patch_states, patch_actions):
                    in_vec = torch.zeros(32, device=device)
                    in_vec[s_idx] = 1.0
                    collected_inputs.append(in_vec.unsqueeze(0))
                    collected_targets.append(a_idx)
                    
                    # Advance group state based on teacher advice
                    state_vector = torch.matmul(generators[a_idx], state_vector)
                    env.process_direct_index(torch.argmax(state_vector).item())
                continue
                
            # Default operation under trained network guidance
            with torch.no_grad():
                logits = router(state_vector)
                chosen_action = torch.argmax(logits).item()
                
            # Keep trace for standard self-imitation updating
            collected_inputs.append(state_vector.unsqueeze(0))
            collected_targets.append(chosen_action)
            
            state_vector = torch.matmul(generators[chosen_action], state_vector)
            env.process_direct_index(torch.argmax(state_vector).item())
            
        # Supervised optimization pass aggregating structural steps and SMT repair steps
        if len(collected_targets) > 0:
            inputs_tensor = torch.cat(collected_inputs, dim=0)
            targets_tensor = torch.tensor(collected_targets, dtype=torch.long, device=device)
            
            logits = router(inputs_tensor)
            loss = loss_fn(logits, targets_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        score = env.get_filled_orders_count()
        performance_curve.append(score)
        
        if (iteration + 1) % 15 == 0:
            print(f"  Iteration {iteration+1:03d} -> Group Scheduler Dynamic Success Rate: {score}/4 Tables Managed")
            
    # Export the co-evolution milestone tracking plot
    plt.figure(figsize=(10, 5))
    plt.plot(performance_curve, color='teal', linewidth=2, label='Co-Evolutionary Group Net')
    plt.title('Step 3: Emergent Scheduling Resilience via SMT Remediation')
    plt.xlabel('Co-Evolutionary Iterations')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('coevolution_improvement_curve.png')
    print("\n📊 PLOT EXPORTED SUCCESS: Saved as 'coevolution_improvement_curve.png'")

if __name__ == "__main__":
    run_step3_coevolution_experiment()

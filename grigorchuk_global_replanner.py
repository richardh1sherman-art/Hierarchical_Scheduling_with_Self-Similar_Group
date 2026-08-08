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
# NEURAL ROUTER
# ==========================================
class GrigorchukGlobalRouter(torch.nn.Module):
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
# DYNAMIC ENVIRONMENT & UNIFIED GLOBAL SMT REPLANNER
# ==========================================================
class GlobalReplanningEnvironment:
    def __init__(self, compiler):
        self.compiler = compiler
        self.reset()
        self.transition_map = self._precompute_matrix_transitions()

    def reset(self):
        self.table_stages = np.zeros(4, dtype=int)
        
    def _precompute_matrix_transitions(self):
        m_map = {0: {}, 1: {}, 2: {}, 3: {}}
        mats = [self.compiler.a_mat, self.compiler.b_mat, self.compiler.c_mat, self.compiler.d_mat]
        for m_idx, mat in enumerate(mats):
            for src in range(32):
                vec = torch.zeros(32, device=device)
                vec[src] = 1.0
                out_vec = torch.matmul(mat, vec)
                dst = torch.argmax(out_vec).item()
                m_map[m_idx][src] = dst
        return m_map

    def inject_realtime_shock(self, step_tick):
        if step_tick == 12 and random.random() < 0.60:
            target_table = random.randint(0, 3)
            self.table_stages[target_table] = 0  
            return True
        return False

    def calculate_global_replanning_trace(self, current_index, remaining_horizon=15):
        opt = Optimize()
        states = [Int(f"r_state_{t}") for t in range(remaining_horizon + 1)]
        actions = [Int(f"r_act_{t}") for t in range(remaining_horizon)]
        
        opt.add(states == current_index)
        
        for t in range(remaining_horizon):
            opt.add(actions[t] >= 0, actions[t] <= 3)
            
            z3_transition = states[t]
            for m_idx in range(4):
                condition = (actions[t] == m_idx)
                branch_logic = states[t]
                for src_idx, dst_idx in self.transition_map[m_idx].items():
                    branch_logic = If(states[t] == src_idx, dst_idx, branch_logic)
                z3_transition = If(condition, branch_logic, z3_transition)
                
            opt.add(states[t+1] == z3_transition)
            
        opt.add(states[remaining_horizon] == 15)
        
        if opt.check() == sat:
            m = opt.model()
            return [m[actions[t]].as_long() for t in range(remaining_horizon)], [m[states[t]].as_long() for t in range(remaining_horizon)]
        
        return [3] * remaining_horizon, [current_index] * remaining_horizon

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
# EXPERIMENTAL TRAINER FUNCTION
# ==========================================
def run_global_replanning_experiment(iterations=150, horizon=30):
    print("====================================================================")
    print("🔬 STEPS 2&3: GLOBAL RE-SCHEDULING ENVIRONMENT (TRUE KRONECKER REPLAN)")
    print("====================================================================\n")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    env = GlobalReplanningEnvironment(compiler)
    router = GrigorchukGlobalRouter(dim=32).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=0.005)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    performance_curve = []
    
    for iteration in range(iterations):
        env.reset()
        
        # FIXED: Explicit one-hot indexing preserves PyTorch Tensor structure
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0  
        
        collected_inputs = []
        collected_targets = []
        
        current_action_plan, _ = env.calculate_global_replanning_trace(0, remaining_horizon=horizon)
        plan_pointer = 0
        
        for step in range(horizon):
            curr_idx = torch.argmax(state_vector).item()
            
            world_changed = env.inject_realtime_shock(step)
            if world_changed:
                if iteration % 30 == 0:
                    print(f"    [Iteration {iteration+1:03d} | Step {step:02d}] ⚠️ ANOMALY! Scrapping schedule. Running global Group Replanner...")
                
                remaining_steps = horizon - step
                current_action_plan, _ = env.calculate_global_replanning_trace(curr_idx, remaining_horizon=remaining_steps)
                plan_pointer = 0
            
            if plan_pointer < len(current_action_plan):
                chosen_action = current_action_plan[plan_pointer]
                plan_pointer += 1
            else:
                chosen_action = 3 
                
            collected_inputs.append(state_vector.unsqueeze(0))
            collected_targets.append(chosen_action)
            
            state_vector = torch.matmul(generators[chosen_action], state_vector)
            env.process_direct_index(torch.argmax(state_vector).item())
            
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
            print(f"  Iteration {iteration+1:03d} -> Global Replanner Success Rate: {score}/4 Tables Managed")
            
    plt.figure(figsize=(10, 5))
    plt.plot(performance_curve, color='navy', linewidth=2, label='Global Replanning Group Net')
    plt.title('Step 3: Scheduling Efficiency under Complete Global Replanning')
    plt.xlabel('Training Iterations')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('global_replanning_curve.png')
    print("\n📊 REPLANNING BENCHMARK COMPLETE: Graph output to 'global_replanning_curve.png'")

if __name__ == "__main__":
    run_global_replanning_experiment()

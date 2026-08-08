import torch
import numpy as np
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ALGEBRAIC LAYER: YOUR GRIGORCHUK COMPILER
# ==========================================
class TrueBottomUpGrigorchukCompiler:
    def __init__(self, max_depth=7):
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

# ==========================================================
# REACTIVE PURE GROUP ENVIRONMENT (DEPTH 7 EXTENSION)
# ==========================================================
class PureGroupEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        # 4 tables tracking stages: 0=Idle, 1=Ordered, 2=Coffee, 3=Fulfilled
        self.table_stages = np.zeros(4, dtype=int)
        
    def inject_realtime_shock(self, step_tick):
        if step_tick == 12:
            target_table = random.randint(0, 3)
            self.table_stages[target_table] = 0  
            return f"⚠️ ANOMALY: Table {target_table} changed their mind!"
        return None

    def execute_confirmed_index(self, confirmed_idx):
        # Mapped to read safely inside the 128 leaf space (shifting past depth 7)
        table_focus = (confirmed_idx >> 4) & 3
        task_action  = confirmed_idx & 3
        if table_focus >= 4: return
        
        current_stage = self.table_stages[table_focus]
        if task_action == 1 and current_stage == 0: self.table_stages[table_focus] = 1
        elif task_action == 2 and current_stage == 1: self.table_stages[table_focus] = 2
        elif task_action == 3 and current_stage == 2: self.table_stages[table_focus] = 3

    def get_filled_orders_count(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================================
# REVERSIBLE DEEP ALGEBRAIC LOOK-AHEAD ENGINE
# ==========================================================
def run_reversible_pure_scheduler(iterations=30, horizon=25):
    print("====================================================================")
    print("🔮 DEEP HORIZON REVERSIBLE SCHEDULER: COMPILING MAX_DEPTH=7 UNIVERSE")
    print("====================================================================\n")
    
    # Initialize compiler at depth 7 -> 128 Dimensions
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=7)
    env = PureGroupEnvironment()
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    gen_names = ['a', 'b', 'c', 'd']
    
    performance_curve = []
    
    # Target terminal index mapped safely inside the deep 128 layout space
    goal_vector = torch.zeros(compiler.dim, device=device)
    goal_vector[63] = 1.0

    for iteration in range(iterations):
        env.reset()
        
        # Initialize 128-dimensional starting point tensor
        state_vector = torch.zeros(compiler.dim, device=device)
        state_vector[0] = 1.0
        
        trace_log = []
        
        for step in range(horizon):
            shock_alert = env.inject_realtime_shock(step)
            if shock_alert and iteration % 10 == 0:
                print(f"    [Iteration {iteration+1:03d} | Step {step:02d}] {shock_alert}")
                
            best_action_idx = 0
            best_alignment = -9999.0
            best_next_vector = None
            
            for idx, gen in enumerate(generators):
                fwd_vector = torch.matmul(gen, state_vector)
                rev_vector = torch.matmul(gen, goal_vector)
                
                # Bidirectional vector convergence alignment
                alignment = torch.dot(fwd_vector, rev_vector).item()
                
                cand_idx = torch.argmax(fwd_vector).item()
                # Tracking incentive adjusted for Depth 7 dimensions
                if (cand_idx & 3) > 0 and cand_idx < 64:
                    alignment += 5.0
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_action_idx = idx
                    best_next_vector = fwd_vector
            
            state_vector = best_next_vector
            trace_idx = torch.argmax(state_vector).item()
            
            env.execute_confirmed_index(trace_idx)
            trace_log.append(gen_names[best_action_idx])
            
        score = env.get_filled_orders_count()
        performance_curve.append(score)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> Pure Group Algebra Success Rate: {score}/4 Tables Managed")
            print(f"    Executed Algebraic Word Sequence: {' · '.join(trace_log[:12])}...")
            
    plt.figure(figsize=(10, 5))
    plt.plot(performance_curve, color='crimson', linewidth=2)
    plt.title('Depth 7 Algebraic Group Scheduling Performance')
    plt.xlabel('Evaluation Iterations')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('pure_algebraic_performance.png')
    print("\n========= RUN COMPLETE: Depth 7 metric graph output to 'pure_algebraic_performance.png' =========")

if __name__ == "__main__":
    run_reversible_pure_scheduler()

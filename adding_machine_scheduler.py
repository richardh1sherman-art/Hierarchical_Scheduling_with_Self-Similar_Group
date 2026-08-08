import torch
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# ALGEBRAIC LAYER: INDUCTIVE ADDING MACHINE TENSOR COMPILER
# ==========================================================
class TrueAddingMachineCompiler:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.odometer_mat = self._build_adding_layer(max_depth)
        self.dim = 2 ** max_depth

    def _build_adding_layer(self, current_depth):
        if current_depth == 1:
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)

        sub_depth = current_depth - 1
        sub_dim = 2 ** sub_depth
        dim_local = 2 ** current_depth
        sub_odometer = self._build_adding_layer(sub_depth)
        
        odometer_local = torch.zeros(dim_local, dim_local, device=device)
        odometer_local[:sub_dim, :sub_dim] = torch.eye(sub_dim, device=device)
        odometer_local[sub_dim:, sub_dim:] = sub_odometer
        
        permutation_shift = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        perm_mask = torch.kron(permutation_shift, torch.eye(sub_dim, device=device))
        return torch.matmul(perm_mask, odometer_local)

# ==========================================================
# REACTIVE PURE GROUP ENVIRONMENT (WITH AUTOMATIC ALIGNMENT)
# ==========================================================
class PureGroupEnvironment:
    def __init__(self):
        self.reset()
        self.milestone_map = {}

    def reset(self):
        # 4 distinct task goals
        self.table_stages = np.zeros(4, dtype=int)

    def auto_align_vocabulary(self, compiler):
        """
        DIAGNOSTIC: Traces how the 'g' matrix naturally steps from index 0 
        and builds an environment vocabulary tailored to the group's true physics.
        """
        g_mat = compiler.odometer_mat
        state = torch.zeros(32, device=device)
        state[0] = 1.0
        
        print("\n🔍 DIAGNOSTIC ELEMENT TRACE (How 'g' cycles the 32 leaves):")
        sequence = []
        for step in range(32):
            idx = torch.argmax(state).item()
            sequence.append(idx)
            state = torch.matmul(g_mat, state)
            
        print(f"  True Path: {' -> '.join(map(str, sequence))}\n")
        
        # Map the true visited sequence indices to logical tables 0-3 and tasks 1-3
        for task_idx, state_idx in enumerate(sequence[1:13]): # Take first 12 transitions
            table = task_idx // 3
            task = (task_idx % 3) + 1
            self.milestone_map[state_idx] = (table, task)

    def preview_score_gain(self, target_idx, stages_state):
        """Evaluates utility using the aligned group physics map."""
        if target_idx not in self.milestone_map:
            return -0.1
            
        table, task = self.milestone_map[target_idx]
        current_stage = stages_state[table]
        
        if task == 1 and current_stage == 0: return 100.0
        if task == 2 and current_stage == 1: return 200.0
        if task == 3 and current_stage == 2: return 500.0
        return -0.2

    def execute_confirmed_index(self, confirmed_idx):
        if confirmed_idx not in self.milestone_map: return
        table, task = self.milestone_map[confirmed_idx]
        current_stage = self.table_stages[table]
        
        if task == 1 and current_stage == 0: self.table_stages[table] = 1
        elif task == 2 and current_stage == 1: self.table_stages[table] = 2
        elif task == 3 and current_stage == 2: self.table_stages[table] = 3

    def get_filled_orders_count(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================================
# MULTI-STEP WORD LOOK-AHEAD ENGINE
# ==========================================================
def run_adding_machine_experiment(iterations=30, horizon=36, lookahead_depth=3):
    print("====================================================================")
    print(f"🔮 RUNNING DEEP HORIZON ODOMETER: {lookahead_depth}-STEP DIAGNOSTIC SEARCH")
    print("====================================================================\n")
    
    compiler = TrueAddingMachineCompiler(max_depth=5)
    env = PureGroupEnvironment()
    env.auto_align_vocabulary(compiler)  # Run vocabulary calibration
    
    g = compiler.odometer_mat
    g_inv = torch.inverse(g)
    identity = torch.eye(32, device=device)
    
    operators = [g, g_inv, identity]
    op_names = ['g', 'g_inv', 'Hold']
    
    word_sequences = list(itertools.product(range(3), repeat=lookahead_depth))
    performance_curve = []

    for iteration in range(iterations):
        env.reset()
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0  
        
        trace_log = []
        step = 0
        
        while step < horizon:
            curr_idx = torch.argmax(state_vector).item()
            
            best_word = None
            best_word_utility = -99999.0
            best_final_vector = None
            
            for word in word_sequences:
                temp_vector = state_vector.clone()
                temp_stages = env.table_stages.copy()
                word_utility = 0.0
                
                for op_idx in word:
                    temp_vector = torch.matmul(operators[op_idx], temp_vector)
                    cand_idx = torch.argmax(temp_vector).item()
                    word_utility += env.preview_score_gain(cand_idx, temp_stages)
                    
                    # Update local sandbox state if a milestone is cleared
                    if cand_idx in env.milestone_map:
                        t, tk = env.milestone_map[cand_idx]
                        if tk == 1 and temp_stages[t] == 0: temp_stages[t] = 1
                        elif tk == 2 and temp_stages[t] == 1: temp_stages[t] = 2
                        elif tk == 3 and temp_stages[t] == 2: temp_stages[t] = 3
                            
                if word_utility > best_word_utility:
                    best_word_utility = word_utility
                    best_word = word
                    best_final_vector = temp_vector
            
            # Execute winner word
            for op_idx in best_word:
                state_vector = torch.matmul(operators[op_idx], state_vector)
                trace_log.append(op_names[op_idx])
                final_idx = torch.argmax(state_vector).item()
                env.execute_confirmed_index(final_idx)
                
            step += lookahead_depth
            
        score = env.get_filled_orders_count()
        performance_curve.append(score)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> Aligned Success Rate: {score}/4 Tables Completed")
            print(f"    Discovered Odometer Word Structure: {' · '.join(trace_log[:12])}...")
            
    plt.figure(figsize=(10, 5))
    plt.plot(performance_curve, color='royalblue', linewidth=2)
    plt.title('Aligned Adding Group Scheduling Performance')
    plt.xlabel('Evaluation Iterations')
    plt.ylabel('Completed Table Orders')
    plt.grid(True)
    plt.savefig('adding_machine_performance.png')
    print("\n========= RUN SUCCESSFUL: Performance curve output to 'adding_machine_performance.png' =========")

if __name__ == "__main__":
    run_deep_algebraic_experiment = run_adding_machine_experiment # Aliasing to preserve runner handle
    run_deep_algebraic_experiment()

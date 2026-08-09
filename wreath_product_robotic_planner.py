import torch
import numpy as np
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LEVEL 2 ALGEBRA: THE LOCAL ROBOTIC ACTUATOR (GRIGORCHUK)
# ==========================================================
class RoboticGrigorchukCompiler:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.dim = 2 ** max_depth
        self.a_mat, self.b_mat, self.c_mat, self.d_mat = self._build_actuator_layer(max_depth)

    def _build_actuator_layer(self, current_depth):
        if current_depth == 1:
            a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
            b, c, d = torch.eye(2, device=device), torch.eye(2, device=device), torch.eye(2, device=device)
            return a, b, c, d

        sub_depth = current_depth - 1
        sub_dim = 2 ** sub_depth
        dim_local = 2 ** current_depth
        
        a_sub, b_sub, c_sub, d_sub = self._build_actuator_layer(sub_depth)
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
# LEVEL 1 ALGEBRA: THE CENTRAL DISPATCHER (ADDING MACHINE)
# ==========================================================
class CentralOdometerCompiler:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.dim = 2 ** max_depth
        self.g_mat = self._build_dispatch_layer(max_depth)

    def _build_dispatch_layer(self, current_depth):
        if current_depth == 1:
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)

        sub_depth = current_depth - 1
        sub_dim = 2 ** sub_depth
        dim_local = 2 ** current_depth
        sub_odometer = self._build_dispatch_layer(sub_depth)
        
        odometer_local = torch.zeros(dim_local, dim_local, device=device)
        odometer_local[:sub_dim, :sub_dim] = torch.eye(sub_dim, device=device)
        odometer_local[sub_dim:, sub_dim:] = sub_odometer
        
        permutation_shift = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
        perm_mask = torch.kron(permutation_shift, torch.eye(sub_dim, device=device))
        return torch.matmul(perm_mask, odometer_local)

# ==========================================================
# PHYSICAL ENVIRONMENT: INDUSTRIAL ROBOT REGULATOR
# ==========================================================
class RobotFleetEnvironment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.table_stages = np.zeros(4, dtype=int)
        
    def preview_step_utility(self, assigned_robot, mechanical_action, stages_state):
        current_stage = stages_state[assigned_robot]
        if mechanical_action == 1 and current_stage == 0: return 100.0
        if mechanical_action == 2 and current_stage == 1: return 300.0
        if mechanical_action == 3 and current_stage == 2: return 1000.0
        return -0.1

    def execute_confirmed_index(self, assigned_robot, mechanical_action):
        current_stage = self.table_stages[assigned_robot]
        token = "System-Idle"
        
        if mechanical_action == 1 and current_stage == 0:
            self.table_stages[assigned_robot] = 1
            token = f"Robot_{assigned_robot}: Lock-On Target (Order)"
        elif mechanical_action == 2 and current_stage == 1:
            self.table_stages[assigned_robot] = 2
            token = f"Robot_{assigned_robot}: Actuate Gripper (Coffee)"
        elif mechanical_action == 3 and current_stage == 2:
            self.table_stages[assigned_robot] = 3
            token = f"Robot_{assigned_robot}: Deploy Cargo (Meal)"
            
        return token

    def get_fleet_efficiency(self):
        return int(np.sum(self.table_stages == 3))

# ==========================================================
# THE HIERARCHICAL WREATH PRODUCT PLANNER ENGINE
# ==========================================================
def execute_wreath_product_planner(horizon=120, lookahead_depth=5):
    print("====================================================================")
    print(f"🤖 DEEP WREATH PLANNER: DEPTH {lookahead_depth} WORD LOOK-AHEAD SEARCH")
    print("====================================================================\n")
    
    dispatcher = CentralOdometerCompiler(max_depth=3) 
    actuator = RoboticGrigorchukCompiler(max_depth=3)   
    env = RobotFleetEnvironment()
    
    g_dispatch = dispatcher.g_mat
    actuator_ops = [actuator.a_mat, actuator.b_mat, actuator.c_mat, actuator.d_mat]
    actuator_names = ['a', 'b', 'c', 'd']
    
    # Precompute deep word strings (4^5 = 1024 candidates)
    word_sequences = list(itertools.product(range(4), repeat=lookahead_depth))
    
    # FIXED: True Tensor assignment via indexing
    dispatch_vector = torch.zeros(8, device=device)
    dispatch_vector[0] = 1.0
    
    # FIXED: True Tensor assignment via indexing inside the dictionary loop
    robot_registers = {}
    for r_id in range(4):
        v = torch.zeros(8, device=device)
        v[0] = 1.0  
        robot_registers[r_id] = v
    
    print("⚡ Synthesizing Generative Algebraic Action Plan...\n")
    step = 0
    
    while step < horizon:
        current_dispatch_idx = torch.argmax(dispatch_vector).item()
        assigned_robot = (current_dispatch_idx // 2) % 4
        
        active_actuator_vector = robot_registers[assigned_robot]
        
        best_word = None
        best_word_utility = -99999.0
        best_final_actuator_vector = None
        
        for word in word_sequences:
            temp_actuator_vector = active_actuator_vector.clone()
            temp_stages = env.table_stages.copy()
            word_utility = 0.0
            
            for op_idx in word:
                temp_actuator_vector = torch.matmul(actuator_ops[op_idx], temp_actuator_vector)
                cand_actuator_idx = torch.argmax(temp_actuator_vector).item()
                mechanical_action = cand_actuator_idx % 4
                
                word_utility += env.preview_step_utility(assigned_robot, mechanical_action, temp_stages)
                
                curr_stg = temp_stages[assigned_robot]
                if mechanical_action == 1 and curr_stg == 0: temp_stages[assigned_robot] = 1
                elif mechanical_action == 2 and curr_stg == 1: temp_stages[assigned_robot] = 2
                elif mechanical_action == 3 and curr_stg == 2: temp_stages[assigned_robot] = 3
                
            if word_utility > best_word_utility:
                best_word_utility = word_utility
                best_word = word
                best_final_actuator_vector = temp_actuator_vector
                
        # Advance timeline log during stasis alignment skips
        if best_word_utility <= 0.0:
            print(f"  Step {step+1:02d} | Master Pulse: Odometer -> Index {current_dispatch_idx:02d} | Actuator: Hold | System-Stasis Alignment")
            dispatch_vector = torch.matmul(g_dispatch, dispatch_vector)
            step += 1
            continue
            
        # Execute the deeper 5-step macro-word for the selected robot
        for op_idx in best_word:
            robot_registers[assigned_robot] = torch.matmul(actuator_ops[op_idx], robot_registers[assigned_robot])
            current_actuator_idx = torch.argmax(robot_registers[assigned_robot]).item()
            mechanical_action = current_actuator_idx % 4
            
            log_token = env.execute_confirmed_index(assigned_robot, mechanical_action)
            print(f"  Step {step+1:02d} | Master Pulse: Odometer -> Index {current_dispatch_idx:02d} | Actuator: {actuator_names[op_idx]} | {log_token}")
            
            step += 1
            if step >= horizon: break
            
        dispatch_vector = torch.matmul(g_dispatch, dispatch_vector)
            
    print(f"\n🏆 Task Complete: Centralized Planner successfully secured {env.get_fleet_efficiency()}/4 Robotic Stations!")

if __name__ == "__main__":
    execute_wreath_product_planner()

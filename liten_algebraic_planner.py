import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LITEN BASE INTERFACE MOCK
# ==========================================================
class BaseWorldStub:
    def __init__(self):
        pass
    def reset(self):
        raise NotImplementedError
    def act(self, action):
        raise NotImplementedError

# ==========================================================
# ALGEBRAIC CORE: MEALY AUTOMATON ODOMETER
# ==========================================================
class MealyOdometerAutomaton:
    def __init__(self):
        self.state = 0

    def step(self, input_bit):
        if self.state == 0:
            if input_bit == 0:
                output_bit = 1
                self.state = 0
            else:
                output_bit = 0
                self.state = 1
        else:
            if input_bit == 0:
                output_bit = 0
                self.state = 1
            else:
                output_bit = 1
                self.state = 0
        return output_bit

# ==========================================================
# LITEN INTERFACE: ALGEBRAIC PLANNER WORLD STUB
# ==========================================================
class AlgebraicPlannerStub(BaseWorldStub):
    def __init__(self, tree_depth=5):
        super().__init__()
        self.tree_depth = tree_depth
        self.automaton = MealyOdometerAutomaton()
        self.reset()

    def reset(self):
        self.automaton.state = 0
        self.robot_stages = np.zeros(4, dtype=int)
        print("🤖 [LITEN STUB] Robot fleet configurations reset to ground zero.")
        return {"status": "initialized", "stages": self.robot_stages.copy()}

    def act(self, action_string):
        self.automaton.state = 0  
        output_stream = []
        
        for char in action_string:
            bit = int(char)
            out_bit = self.automaton.step(bit)
            output_stream.append(out_bit)
            
        resultant_index = 0
        for power, bit in enumerate(reversed(output_stream)):
            resultant_index += bit * (2 ** power)
            
        assigned_robot = (resultant_index // 2) % 4
        mechanical_action = resultant_index % 4
        
        current_stage = self.robot_stages[assigned_robot]
        action_token = f"Robot_{assigned_robot}: Hold-Position"
        is_valid_milestone = False
        
        if mechanical_action == 1 and current_stage == 0:
            self.robot_stages[assigned_robot] = 1
            action_token = f"Robot_{assigned_robot} Executed: LOCK-ON TARGET (Order)"
            is_valid_milestone = True
        elif mechanical_action == 2 and current_stage == 1:
            self.robot_stages[assigned_robot] = 2
            action_token = f"Robot_{assigned_robot} Executed: ACTUATE GRIPPER (Coffee)"
            is_valid_milestone = True
        elif mechanical_action == 3 and current_stage == 2:
            self.robot_stages[assigned_robot] = 3
            action_token = f"Robot_{assigned_robot} Executed: DEPLOY CARGO (Meal)"
            is_valid_milestone = True
            
        return {
            "execution_token": action_token,
            "fleet_status": self.robot_stages.copy(),
            "valid_milestone": is_valid_milestone
        }

# ==========================================================
# STEP 1.1 & 1.2: HIERARCHICAL ANALYSIS & LOSS CALCULATOR
# ==========================================================
class LitenHistoryCompiler:
    def __init__(self, stub):
        self.stub = stub
        self.task_generator_map = {
            "pick_up_block": "010",
            "align_gripper": "110",
            "secure_object": "011",
            "move_to_bin":   "101",
        }
        # Initialize an algebraic penalty accumulation tracker
        self.total_algebraic_loss = 0.0

    def execute_logged_trace(self, trace_string, simulated_liten_assessments):
        print(f"\n📂 [STEP 1.1] Parsing Ingested LITEN Execution Trace Log...")
        log_tokens = trace_string.strip().split(" -> ")
        
        for step_idx, token in enumerate(log_tokens):
            if token in self.task_generator_map:
                binary_pulse = self.task_generator_map[token]
                feedback = self.stub.act(binary_pulse)
                
                # Fetch LITEN's binary vision-language feedback signal for this specific step
                vlm_success_signal = simulated_liten_assessments.get(token, True)
                
                # STEP 1.2 REPURPOSING LOGIC:
                # If the state machine says it's a structural milestone, but the VLM flags a 
                # physical execution failure, apply a massive algebraic loss weight penalty!
                loss_weight = 0.0
                if feedback["valid_milestone"] and not vlm_success_signal:
                    loss_weight = 500.0  # High penalty weight for unaligned paths
                    self.total_algebraic_loss += loss_weight
                elif not feedback["valid_milestone"]:
                    loss_weight = 10.0   # Minor structural stasis cost
                    self.total_algebraic_loss += loss_weight
                    
                status_str = "SUCCESS" if vlm_success_signal else "FAILURE (Collision/Miss)"
                print(f"  Step {step_idx+1:02d} | Task: {token:<15} ➔ VLM: {status_str:<22} ➔ Loss Weight: {loss_weight:>5.1f} ➔ {feedback['execution_token']}")
            else:
                print(f"  Step {step_idx+1:02d} | Warning: Unknown token '{token}' skipped.")

if __name__ == "__main__":
    print("====================================================================")
    print("🔬 TESTING STEP 1.2: LITEN VLM ASSESSMENT REPURPOSING LOSS MODULE")
    print("====================================================================\n")
    
    liten_stub = AlgebraicPlannerStub(tree_depth=5)
    history_compiler = LitenHistoryCompiler(liten_stub)
    
    simulated_liten_log = "pick_up_block -> align_gripper -> secure_object -> move_to_bin"
    
    # Simulating LITEN's assessment logs where 'secure_object' suffered a physical gripper slippage/failure
    simulated_vlm_feedback = {
        "pick_up_block": True,
        "align_gripper": True,
        "secure_object": False,  # VLM flagged a physical failure here!
        "move_to_bin":   True
    }
    
    # Run the adaptive loss calculation loop
    history_compiler.execute_logged_trace(simulated_liten_log, simulated_vlm_feedback)
    
    print("\n📊 ================================================================")
    print(f"🏆 Trajectory Evaluation Loss Generated: {history_compiler.total_algebraic_loss} Units")
    print("====================================================================")

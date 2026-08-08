import torch
import time

# Use the DGX GPU backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralGroupPolicy(torch.nn.Module):
    """
    A real PyTorch neural network that takes the current restaurant state
    and outputs probabilities for choosing generator G1 vs G2.
    """
    def __init__(self):
        super().__init__()
        # Input: 32-bit state representation unrolled or embedded
        self.fc = torch.nn.Linear(32, 2) 
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state_bits):
        # Convert integer bits to a float tensor for the network
        x = state_bits.float()
        logits = self.fc(x)
        return self.softmax(logits)

def unpack_state(state):
    """Unpacks a 32-bit integer state into individual environment variables."""
    w1_loc   = (state >> 0) & 7
    w2_loc   = (state >> 3) & 7
    w1_hand  = (state >> 6) & 7
    w2_hand  = (state >> 9) & 7
    orders   = (state >> 12) & 31
    return w1_loc, w2_loc, w1_hand, w2_hand, orders

# --- THE SELF-SIMILAR GROUP ALGEBRA GENERATORS ---

def G1_MicroAction(state):
    """Generator G1: Local state mutation (Finite State Grammar step)."""
    w1_loc, w2_loc, w1_hand, w2_hand, orders = unpack_state(state)
    
    # Micro-law: If at kitchen (0), collect coffee (1). If at a table (>0), deliver it.
    if w1_loc == 0:
        new_w1_hand = 1
    else:
        new_w1_hand = 0
        if w1_hand == 1:
            orders |= (1 << (w1_loc - 1)) # Mark table order as filled
            
    # Repack the modified algebraic state
    clear_mask = ~((7 << 6) | (31 << 12))
    new_state = (state & clear_mask) | (new_w1_hand << 6) | (orders << 12)
    return new_state

def G2_TableSwitch(state):
    """Generator G2: Structural rotation/context switch across the tree hierarchy."""
    w1_loc, w2_loc, w1_hand, w2_hand, orders = unpack_state(state)
    
    # Macro-law: Cycle waitresses to the next tables
    new_w1_loc = (w1_loc + 1) % 6
    new_w2_loc = (w2_loc + 2) % 6
    
    clear_mask = ~7 & ~(7 << 3)
    new_state = (state & clear_mask) | new_w1_loc | (new_w2_loc << 3)
    return new_state

# --- SIMULATION LOOP ---

def run_pure_group_scheduler(steps=15):
    print("=== STEP 1: INITIALIZING PURE ALGEBRAIC GROUP SCHEDULER ===")
    print(f"Target Hardware: {device}\n")

    policy_net = NeuralGroupPolicy().to(device)
    
    # Initial state: everything at 0 (Kitchen, empty hands, unfilled orders)
    current_state = 0 
    state_history = [current_state]
    actions_taken = []

    for step in range(steps):
        # 1. Convert current scalar state to a mock binary tensor for the NN
        state_bits = [int(x) for x in f"{current_state:032b}"]
        state_tensor = torch.tensor(state_bits, dtype=torch.float32).to(device)
        
        # 2. NN predicts the group action
        action_probs = policy_net(state_tensor)
        chosen_generator = torch.argmax(action_probs).item()
        
        # 3. Apply the Group Algebraic operators directly
        if chosen_generator == 0:
            current_state = G1_MicroAction(current_state)
            actions_taken.append("G1 (Micro-Action)")
        else:
            current_state = G2_TableSwitch(current_state)
            actions_taken.append("G2 (Table Switch)")
            
        state_history.append(current_state)

    # Output results
    print("Executed Group Sequence Trace:")
    for i, act in enumerate(actions_taken):
        w1_l, _, w1_h, _, ords = unpack_state(state_history[i])
        print(f"  Step {i:02d}: W1 at Table {w1_l}, Holding {w1_h}, Orders Filled Mask: {ords:05b} -> Chose {act}")
        
    _, _, _, _, final_orders = unpack_state(current_state)
    print(f"\nFinal Achievement Score: {bin(final_orders).count('1')}/5 Orders Filled")

if __name__ == "__main__":
    run_pure_group_scheduler()

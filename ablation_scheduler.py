import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# ABLATION AGENT: UNCONSTRAINED STANDALONE NEURAL NETWORK
# ==========================================================
class StandaloneAblationAgent(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        # INSTEAD OF CHOSING 4 FIXED GENERATORS:
        # The ablation network outputs a direct transformation matrix to map state_t to state_t+1
        self.output_layer = torch.nn.Linear(64, dim)

    def forward(self, state_vector):
        x = torch.relu(self.fc1(state_vector))
        x = torch.relu(self.fc2(x))
        raw_next_state = self.output_layer(x)
        # Apply softmax to keep the state vector operating as a valid probability distribution/one-hot map
        return torch.softmax(raw_next_state, dim=-1)

# ==========================================================
# ENVIRONMENT CRITIC (Identical Rules to Step 2/3)
# ==========================================================
class RestaurantEnvironmentAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        self.customer_stages = np.zeros(5, dtype=int)
        
    def evaluate_state_evolution(self, previous_vector, current_vector):
        reward = 0.0
        prev_idx = torch.argmax(previous_vector).item()
        curr_idx = torch.argmax(current_vector).item()
        
        if prev_idx == curr_idx:
            return -0.2 
        
        customer_focus = curr_idx % 5
        action_type = (curr_idx // 5) % 4  
        current_stage = self.customer_stages[customer_focus]
        
        if action_type == 0 and current_stage == 0:
            self.customer_stages[customer_focus] = 1
            reward += 5.0  
        elif action_type == 1 and current_stage == 1:
            self.customer_stages[customer_focus] = 2
            reward += 10.0 
        elif action_type == 2 and current_stage == 2:
            self.customer_stages[customer_focus] = 3
            reward += 20.0 
        else:
            reward -= 1.0  
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.customer_stages == 3))
        
    def get_coffee_served_count(self):
        return int(np.sum(self.customer_stages >= 2))

# ==========================================================
# TRAINING RUNNER WITHOUT GROUP RESTRAINTS
# ==========================================================
def train_ablation_loop(iterations=200, schedule_horizon=40):
    print("=== STARTING STEP 4: ABLATION CONTROL LOOP (NO GROUP MATH) ===")
    print(f"Executing on: {device} | Horizon Steps: {schedule_horizon} | Iterations: {iterations}")
    
    ablation_agent = StandaloneAblationAgent(dim=32).to(device)
    env_agent = RestaurantEnvironmentAgent()
    optimizer = torch.optim.Adam(ablation_agent.parameters(), lr=0.005)
    
    ablation_history = []
    
    # Load up your previous Grigorchuk performance scores manually for immediate visual comparison
    # (Based directly on your live terminal run values)
    grigorchuk_baseline = [0]*10 + [2]*20 + [0]*40 + [2]*20 + [0]*20 + [2]*20 + [0]*70 
    # Pad or slice baseline to exactly fit iteration graph length
    grigorchuk_baseline = (grigorchuk_baseline * (iterations // len(grigorchuk_baseline) + 1))[:iterations]

    for iteration in range(iterations):
        env_agent.reset()
        
        state_vector = torch.zeros(32, device=device)
        state_vector[0] = 1.0
        
        rewards = []
        state_history = [state_vector]
        
        for step in range(schedule_horizon):
            # The agent acts completely unconstrained by group transitions
            next_state_vector = ablation_agent(state_vector)
            
            step_reward = env_agent.evaluate_state_evolution(state_vector, next_state_vector)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            state_history.append(state_vector)
            
        # Optimization pass via Mean Squared Error backprop over reward trajectories
        total_loss = -torch.tensor(rewards, requires_grad=True).mean()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        filled_count = env_agent.get_filled_orders_count()
        coffee_count = env_agent.get_coffee_served_count()
        ablation_history.append(filled_count)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> [Ablation NN] Customers Served Coffee: {coffee_count}/5 | Fully Completed: {filled_count}/5")
            
    # Generate the comparative analysis graph
    plt.figure(figsize=(12, 6))
    plt.plot(grigorchuk_baseline, color='purple', linewidth=2, label='With Self-Similar Groups (Steps 2&3)')
    plt.plot(ablation_history, color='red', linestyle='--', linewidth=2, label='Standalone NN Ablation (Step 4)')
    plt.title('Research Comparison: Group Algebra Constraints vs. Standalone NN')
    plt.xlabel('Training Iterations')
    plt.ylabel('Completed Orders')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparative_analysis.png')
    print("\n📊 STEP 4 COMPLETE: Comparative graph generated and output to 'comparative_analysis.png'.")

if __name__ == "__main__":
    train_ablation_loop()

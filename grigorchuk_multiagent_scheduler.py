import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CORE ALGEBRA: YOUR GRIGORCHUK COMPILER
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
# AGENT 1: SCHEDULING NEURAL NETWORK
# ==========================================
class GrigorchukSchedulingAgent(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 4)  # Generates logits for [a, b, c, d]
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state_vector):
        x = torch.relu(self.fc1(state_vector))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

# ==========================================
# AGENT 2: EXTERNAL ENVIRONMENT CRITIC
# ==========================================
class RestaurantEnvironmentAgent:
    def __init__(self):
        self.reset()

    def reset(self):
        # 5 customers: 0=start, 1=ordered, 2=got_coffee, 3=got_meal (fulfilled)
        self.customer_stages = np.zeros(5, dtype=int)
        
    def evaluate_state_evolution(self, previous_vector, current_vector):
        reward = 0.0
        
        prev_idx = torch.argmax(previous_vector).item()
        curr_idx = torch.argmax(current_vector).item()
        
        if prev_idx == curr_idx:
            return -0.2 # Penalty for idle cycles
        
        # Map the 32 leaves down to individual actions and customers
        customer_focus = curr_idx % 5
        action_type = (curr_idx // 5) % 4  # 0=Take Order, 1=Serve Coffee, 2=Serve Meal, 3=Idle
        
        current_stage = self.customer_stages[customer_focus]
        
        # Rule 1: Take Order
        if action_type == 0 and current_stage == 0:
            self.customer_stages[customer_focus] = 1
            reward += 5.0  # High reward incentive for breaking out of zero state
            
        # Rule 2: Serve Coffee (Order must exist)
        elif action_type == 1 and current_stage == 1:
            self.customer_stages[customer_focus] = 2
            reward += 10.0 
            
        # Rule 3: Serve Meal (Coffee must exist)
        elif action_type == 2 and current_stage == 2:
            self.customer_stages[customer_focus] = 3
            reward += 20.0 
        else:
            reward -= 1.0  # Penalty for invalid out-of-order execution
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.customer_stages == 3))
        
    def get_coffee_served_count(self):
        return int(np.sum(self.customer_stages >= 2))

# ==========================================
# ITERATIVE MULTI-AGENT RUNNER LOOP
# ==========================================
def train_hierarchical_loop(iterations=200, schedule_horizon=40):
    print("=== STARTING STEPS 2 & 3: MULTI-AGENT GRIGORCHUK LOOP ===")
    print(f"Executing on: {device} | Horizon Steps: {schedule_horizon} | Iterations: {iterations}")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    scheduler_agent = GrigorchukSchedulingAgent(dim=compiler.dim).to(device)
    env_agent = RestaurantEnvironmentAgent()
    
    optimizer = torch.optim.Adam(scheduler_agent.parameters(), lr=0.005)
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    history_scores = []
    
    for iteration in range(iterations):
        env_agent.reset()
        
        # Fixed: Initialize as a legitimate 32-dimensional one-hot tensor
        state_vector = torch.zeros(compiler.dim, device=device)
        state_vector[0] = 1.0
        
        log_probabilities = []
        rewards = []
        
        for step in range(schedule_horizon):
            action_probs = scheduler_agent(state_vector)
            
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            log_probabilities.append(action_dist.log_prob(action))
            
            # Apply group matrix transformation directly
            next_state_vector = torch.matmul(generators[action.item()], state_vector)
            
            step_reward = env_agent.evaluate_state_evolution(state_vector, next_state_vector)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            
        # Optimization baseline calculation
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.95 * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards, device=device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        loss = 0
        for log_prob, reward in zip(log_probabilities, discounted_rewards):
            loss -= log_prob * reward
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        filled_count = env_agent.get_filled_orders_count()
        coffee_count = env_agent.get_coffee_served_count()
        history_scores.append(filled_count)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> Customers Served Coffee: {coffee_count}/5 | Fully Completed: {filled_count}/5")
            
    plt.figure(figsize=(10, 5))
    plt.plot(history_scores, color='purple', linewidth=2, label='Grigorchuk Group Scheduler')
    plt.title('Restaurant Order Fulfillment Curve via Algebraic Groups')
    plt.xlabel('Training Iterations')
    plt.ylabel('Fully Completed Orders')
    plt.grid(True)
    plt.savefig('scheduling_performance.png')
    print("\n📊 Done! The revised visualization has been output to 'scheduling_performance.png'.")

if __name__ == "__main__":
    train_hierarchical_loop()

import torch
import numpy as np
import matplotlib.pyplot as plt

# Use GPU fallback safely
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
        self.fc1 = torch.nn.Linear(dim, 64)
        self.fc2 = torch.nn.Linear(64, 4)  # Outputs logits for generators [a, b, c, d]
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state_vector):
        x = torch.relu(self.fc1(state_vector))
        return self.softmax(self.fc2(x))

# ==========================================
# AGENT 2: EXTERNAL ENVIRONMENT CRITIC
# ==========================================
class RestaurantEnvironmentAgent:
    """
    Enforces hidden rules:
    1. Order must be given before anything is served.
    2. Coffee must be served before the meal.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # 5 customers tracking stages: 0=start, 1=ordered, 2=got_coffee, 3=got_meal (fulfilled)
        self.customer_stages = np.zeros(5, dtype=int)
        
    def evaluate_state_evolution(self, previous_vector, current_vector):
        """Maps vector space changes to rule compliance and returns an incremental score."""
        reward = 0.0
        
        # Detect which indices moved in the 32-dimensional leaf space
        prev_idx = torch.argmax(previous_vector).item()
        curr_idx = torch.argmax(current_vector).item()
        
        if prev_idx == curr_idx:
            return -0.1 # Penalty for deadlocks or idle loops
        
        # Map the 32 dimensions down to individual customer state changes
        customer_focus = curr_idx % 5
        action_type = (curr_idx // 5) % 4 # 0=Take Order, 1=Serve Coffee, 2=Serve Meal, 3=Idle
        
        current_stage = self.customer_stages[customer_focus]
        
        # Enforce Hidden Rule 1: Must give order first
        if action_type == 0 and current_stage == 0:
            self.customer_stages[customer_focus] = 1
            reward += 1.0  # Order taken successfully
            
        # Enforce Hidden Rule 2: Coffee before meal
        elif action_type == 1 and current_stage == 1:
            self.customer_stages[customer_focus] = 2
            reward += 2.0  # Coffee served perfectly
            
        elif action_type == 2 and current_stage == 2:
            self.customer_stages[customer_focus] = 3
            reward += 5.0  # Entire order fulfilled!
        else:
            reward -= 0.5  # Out of sequence action penalty
            
        return reward

    def get_filled_orders_count(self):
        return int(np.sum(self.customer_stages == 3))

# ==========================================
# STEPS 2 & 3: ITERATIVE MULTI-AGENT LOOP
# ==========================================
def train_hierarchical_loop(iterations=100, schedule_horizon=20):
    print("=== STARTING STEPS 2 & 3: MULTI-AGENT GRIGORCHUK LOOP ===")
    
    compiler = TrueBottomUpGrigorchukCompiler(max_depth=5)
    scheduler_agent = GrigorchukSchedulingAgent(dim=compiler.dim).to(device)
    env_agent = RestaurantEnvironmentAgent()
    
    optimizer = torch.optim.Adam(scheduler_agent.parameters(), lr=0.01)
    
    # Store matrix references for dynamic lookup
    generators = [compiler.a_mat, compiler.b_mat, compiler.c_mat, compiler.d_mat]
    history_scores = []
    
    for iteration in range(iterations):
        env_agent.reset()
        
        # Initialize scheduling vector as a one-hot vector at the first root leaf
        state_vector = torch.zeros(compiler.dim, device=device)
        state_vector[0] = 1.0
        
        log_probabilities = []
        rewards = []
        
        for step in range(schedule_horizon):
            # 1. Scheduler neural net acts on the algebraic space
            action_probs = scheduler_agent(state_vector)
            
            # Sample action to allow reinforcement exploration
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            log_probabilities.append(action_dist.log_prob(action))
            
            # 2. Apply chosen Grigorchuk operator
            next_state_vector = torch.matmul(generators[action.item()], state_vector)
            
            # 3. Environment Agent checks fulfillment constraints
            step_reward = env_agent.evaluate_state_evolution(state_vector, next_state_vector)
            rewards.append(step_reward)
            
            state_vector = next_state_vector
            
        # Policy gradient update (REINFORCE algorithm step)
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.9 * R
            discounted_rewards.insert(0, R)
            
        discounted_rewards = torch.tensor(discounted_rewards, device=device)
        # Normalize rewards for stable gradient updates
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        loss = 0
        for log_prob, reward in zip(log_probabilities, discounted_rewards):
            loss -= log_prob * reward
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        filled_count = env_agent.get_filled_orders_count()
        history_scores.append(filled_count)
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration+1:03d} -> Orders Filled Correctly: {filled_count}/5")
            
    # Save the execution plot
    plt.figure(figsize=(10, 5))
    plt.plot(history_scores, color='blue', label='Grigorchuk Group Scheduler')
    plt.title('Order Fulfillment Efficiency Over Multi-Agent Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Completed Orders')
    plt.grid(True)
    plt.savefig('scheduling_performance.png')
    print("\n📊 Success! Performance graph saved as 'scheduling_performance.png'.")

if __name__ == "__main__":
    train_hierarchical_loop()

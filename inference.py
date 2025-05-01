import torch
import gym
import numpy as np
from gym.wrappers import TimeLimit
from cartpolechat1 import DQN, add_state_dim  # Assuming DQN is defined in cartpolechat1.py
# --- Load environment ---
env = gym.make("CartPole-v1", render_mode="human")
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env
env = TimeLimit(env, max_episode_steps=3000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Recreate model architecture ---
# DQN should be already defined somewhere like:
# class DQN(nn.Module): ...

state_dim = env.observation_space.shape[0] + 1
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim).to(device)

# --- Load trained weights ---
policy_net.load_state_dict(torch.load("cartpole_best_model.pth", map_location=device))
policy_net.eval()  # Important: Set model to evaluation mode

# --- Inference: 100 episodes ---
total_rewards = []

init_state = [0,0,0,0]

with torch.no_grad():  # No gradients during testing
    for episode in range(1, 101):
        init_state, _ = env.reset()
        init_state = add_state_dim(init_state)
        state = init_state
        env.render()  # Render the environment
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            q_values = policy_net(state_tensor)
            action = q_values.argmax(1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = add_state_dim(next_state)
            
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Test Episode {episode}: State {init_state}: Reward = {total_reward}")
        total_rewards.append(total_reward)

env.close()

# --- Summary ---
print(f"\nAverage Reward over 100 Test Episodes: {sum(total_rewards)/len(total_rewards):.2f}")

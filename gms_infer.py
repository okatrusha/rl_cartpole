import torch
import gymnasium as gym
import numpy as np
import time
from gms_train import DQN, add_state_dim

env = gym.make("CartPole-v1", render_mode="human")
env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=10000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = env.observation_space.shape[0] + 1
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim).to(device)

policy_net.load_state_dict(torch.load("gms_cartpole_best_model.pth", map_location=device))
policy_net.eval()

total_rewards = []

with torch.no_grad():
    for episode in range(1, 101):
        obs, _ = env.reset()
        state = add_state_dim(obs)
        total_reward = 0
        done = False

        while not done:
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = add_state_dim(next_obs)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            if total_reward % 500 == 0:
                print(f"   total reward {total_reward}" )
            # time.sleep(0.01)

        print(f"Test Episode {episode}: Reward = {total_reward}")
        total_rewards.append(total_reward)

env.close()

print(f"\nAverage Reward over 100 Test Episodes: {sum(total_rewards)/len(total_rewards):.2f}")

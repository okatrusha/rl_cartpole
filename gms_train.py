import gymnasium as gym
import numpy as np
# if not hasattr(np, 'bool8'):
#     np.bool8 = np.bool_
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

def add_state_dim(state):
    return np.append(state, state[0] * state[1] * state[2] * state[2] * 10**3)

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1])
        rewards = torch.tensor(batch[2])
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4])

        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def train():
    print(torch.cuda.is_available())
    env = gym.make("CartPole-v1", render_mode=None)
    env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=10000)

    state_dim = env.observation_space.shape[0] + 1
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    buffer = PrioritizedReplayBuffer(20000)

    batch_size = 64
    gamma = 0.99
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 0.9999
    steps_done = 0
    target_update_freq = 200

    all_rewards = []
    epsilon = epsilon_start

    for episode in range(1, 500):
        state, _ = env.reset()
        state = add_state_dim(state)
        total_reward = 0
        done = False

        while not done:
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(state, dtype=torch.float32).to(device)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = add_state_dim(next_state)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1

            if len(buffer.buffer) > batch_size:
                states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones, weights = [x.to(device) for x in [states, actions, rewards, next_states, dones, weights]]

                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1)
                    next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q_values = rewards + gamma * next_q_values * (1 - dones.float())

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                td_errors = q_values - expected_q_values
                loss = (weights * td_errors ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
                buffer.update_priorities(indices, new_priorities)

            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        all_rewards.append(total_reward)
        print(f"Episode {episode} Epsilon {epsilon:.2f} Reward: {total_reward:.0f}")

        if total_reward >= 3000:
            print(f"Solved! Survived 3000 steps at episode {episode}")
            torch.save(policy_net.state_dict(), "gms_cartpole_best_model.pth")
            # break

    plt.plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('CartPole DQN with Prioritized Replay')
    plt.grid()
    plt.show()

    env.close()

if __name__ == "__main__":
    train()

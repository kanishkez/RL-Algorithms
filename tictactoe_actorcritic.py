import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tictactoe_env import TicTacToeEnv
from torch.distributions import Categorical
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)



def select_action(state, actor):
    state_t = torch.tensor(state, dtype=torch.float32)

    probs = actor(state_t)

    # mask invalid actions
    mask = torch.tensor(state == 0, dtype=torch.float32)
    masked_probs = probs * mask

    if masked_probs.sum() == 0:
        masked_probs = mask / mask.sum()

    masked_probs = masked_probs / masked_probs.sum()

    dist = Categorical(masked_probs)
    action = dist.sample()

    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def update(actor, critic,
           actor_optimizer, critic_optimizer,
           log_probs, states, returns):

    states = torch.tensor(states, dtype=torch.float32)

    values = critic(states).squeeze()

    critic_loss = F.mse_loss(values, returns)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    advantage = returns - values.detach()
    actor_loss = -(torch.stack(log_probs) * advantage).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()



def main():
    env = TicTacToeEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    opponent_actor = copy.deepcopy(actor)
    opponent_actor.eval()  # frozen opponent

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    gamma = 0.99
    num_episodes = 5000
    freeze_interval = 200

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        log_probs = []
        rewards = []
        states = []

        while not done:
            if env.current_player == 1:
                # ===== Learning agent =====
                action, log_prob = select_action(state, actor)
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                log_probs.append(log_prob)
                rewards.append(reward)

            else:
                # ===== Frozen opponent =====
                with torch.no_grad():
                    action, _ = select_action(state, opponent_actor)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # If opponent wins → punish learning agent
                if terminated and reward == 1.0:
                    rewards[-1]=-1.0

            done = terminated or truncated
            state = next_state

        if len(states) == 0:
            continue
        returns = compute_returns(rewards, gamma)

        update(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            log_probs,
            states,
            returns
        )

        # ===== Freeze update =====
        if episode % freeze_interval == 0:
            opponent_actor.load_state_dict(actor.state_dict())
            print(f"Episode {episode}: opponent updated")

    torch.save(actor.state_dict(), "tictactoe_actor.pth")
    torch.save(critic.state_dict(), "tictactoe_critic.pth")
    print("Models saved.")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# === Actor-Critic Network ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def act(self, state):
        logits, value = self(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

# === PPO Agent ===
class PPO:
    def __init__(self, state_dim, action_dim, clip_eps=0.2, lr=3e-4):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps

    def compute_loss(self, states, actions, old_log_probs, returns, advantages):
        logits, values = self.policy(states)
        dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
        new_log_probs = dist.log_prob(actions)

        # --- Policy ratio ---
        ratio = torch.exp(new_log_probs - old_log_probs)

        # --- Clipped surrogate objective ---
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # --- Value function loss ---
        value_loss = 0.5 * (returns - values.squeeze())**2
        value_loss = value_loss.mean()

        # --- Entropy bonus (for exploration) ---
        entropy = dist.entropy().mean()

        # --- Total loss ---
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        return loss

    def update(self, data):
        states, actions, old_log_probs, returns, advantages = data
        loss = self.compute_loss(states, actions, old_log_probs, returns, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# === Example Usage ===
if __name__ == "__main__":
    agent = PPO(state_dim=4, action_dim=2)
    batch_size = 8

    # Dummy batch (normally from environment)
    states = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    old_log_probs = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    agent.update((states, actions, old_log_probs, returns, advantages))

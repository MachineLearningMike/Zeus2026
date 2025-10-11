import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# === 1. Simple actor-critic network ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

    def act(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

# === 2. PPO update step ===
def ppo_update(policy, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, batch_size=64):

    for _ in range(epochs):
        for i in range(0, len(states), batch_size):
            batch = slice(i, i + batch_size)
            s, a, old_lp, R, A = states[batch], actions[batch], old_log_probs[batch], returns[batch], advantages[batch]

            logits, values = policy(s)
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()

            new_log_probs = dist.log_prob(a)
            ratio = torch.exp(new_log_probs - old_lp)

            # === Clipped surrogate objective ===
            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
            policy_loss = -torch.min(surr1, surr2).mean()

            # === Value function loss ===
            value_loss = (R - values.squeeze()).pow(2).mean()

            # === Total loss ===
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# === 3. Example usage (conceptual) ===
state_dim, action_dim = 4, 2
policy = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# (Suppose we already collected rollout data)
# states, actions, old_log_probs, returns, advantages = ...

# ppo_update(policy, optimizer, states, actions, old_log_probs, returns, advantages)

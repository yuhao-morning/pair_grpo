import numpy as np

def compute_gae(rewards, dones, values, next_val, gamma, lam):
    advantages = []
    last_adv = 0.0
    for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        delta = r + gamma * next_val * (1 - d) - v
        adv = delta + gamma * lam * (1 - d) * last_adv
        advantages.insert(0, adv)
        last_adv = adv
        next_val = v
    advantages = np.array(advantages)
    returns = advantages + np.array(values)
    return advantages, returns

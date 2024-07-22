from gymnasium import register
from environments.constants import N_ARMS

for n_arms in N_ARMS:
    register(
        id=f"bandits-{n_arms}-v0",
        entry_point="environments.bandits:BanditsEnv",
        max_episode_steps=100,
        kwargs={
            'n_arms': n_arms,
            'state_size': 10,
            'max_steps': 1000
        }
    )
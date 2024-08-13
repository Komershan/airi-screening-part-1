'''
Here we register environments for our experiments and benchmarks

Multi-armed adversarial bandits environment was implemented by myself
Dark room and dark key-to-door environments were borrow from toymeta repository in CORL
https://github.com/corl-team/toy-meta-gym/tree/main/src/toymeta

Here I am register environments only in the configurations that are used in reproduction
'''
from gymnasium import register


'''
Register multi-armed bandits environment
Generation_seed is fictitious because during train and test I always reset environments with fixed sets
For reproducibility purposes

Distribution describes reward distribution in environments, fore more details you can check /environments/bandits.py
'''
for distribution in ["odd", "even", "uniform"]:
    register(
        id=f"bandits-{distribution}-v0",
        entry_point="environments.bandits:BanditsEnv",
        max_episode_steps=100,
        kwargs={
            'n_arms': 10,
            'max_steps': 100,
            'generation_seed': 69,
            'distribution_type': distribution
        },
    )


'''
Here I register Dark Room (light exploration variant) environment
'''        
register(
    id="Dark-Room-v0",
    entry_point="environments.dark_room:DarkRoom",
    max_episode_steps=20,
    kwargs={
        "size": 9,
        "random_start": False,
        "terminate_on_goal": False,
        "goal_only_once": False
    },
)

'''
Here I register Dark Room (hard exploration variant) environment
'''
register(
    id="Dark-Room-Hard-v0",
    entry_point="environments.dark_room:DarkRoom",
    max_episode_steps=20,
    kwargs={
        "size": 17,
        "random_start": False,
        "terminate_on_goal": True,
    },
)

'''
Here I register Dark Key-To-Door environment
'''
register(
    id="Dark-Key-To-Door-9x9-v0",
    entry_point="environments.dark_key_to_door:DarkKeyToDoor",
    max_episode_steps=50,
    kwargs={"size": 9, "random_start": False},
)
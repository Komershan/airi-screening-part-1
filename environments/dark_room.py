'''
Basically, this code is taken from corl-team/toy-meta-gym
https://github.com/corl-team/toy-meta-gym/tree/main

But I added mapping and several methods for environment saving
'''
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.random import Generator, PCG64


def all_goals(grid_size):
    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    return goals


class DarkRoom(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        size=9,
        goal=None,
        random_start=True,
        terminate_on_goal=False,
        render_mode="rgb_array",
        goal_only_once=False,
        seed=69,
    ):
        self.size = size
        self.agent_pos = None
        self.goal_only_once = goal_only_once
        self.was_terminated = False
        self.seed = seed
        # I also fix random generation in order to differentiate tasks
        self.rng = Generator(PCG64(seed=self.seed))

        if goal is not None:
            self.goal_pos = np.asarray(goal)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal()

        self.observation_space = spaces.Discrete(self.size**2)
        self.action_space = spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array((0, 0), dtype=np.float32),  # noop
            1: np.array((-1, 0), dtype=np.float32),  # up
            2: np.array((0, 1), dtype=np.float32),  # right
            3: np.array((1, 0), dtype=np.float32),  # down
            4: np.array((0, -1), dtype=np.float32),  # left
        }

        self.center_pos = (self.size // 2, self.size // 2)
        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode
        self.random_start = random_start

        # I need to have mapping_dict in order to execute environment saving
        # In this dict I map class fields to data types for saving in HDF5
        self.mapping_dict = {
            "size": "int",
            "goal_pos": "array",
            "agent_pos": "array",
            "seed": "int",
            "random_start": "bool",
            "terminate_on_goal": "bool",
            "goal_only_once": "bool",
        }

    def generate_pos(self):
        return self.rng.integers(0, self.size, size=2).astype(np.float32)

    def generate_goal(self):
        return self.rng.integers(0, self.size, size=2)

    def get_action_space_size(self):
        return 5
    
    def get_observation_space_size(self):
        return self.size ** 2

    def set_seed(self, seed=None) -> None:
        self.seed = seed
        self.rng = Generator(PCG64(seed=self.seed))

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    def state_to_pos(self, state):
        return np.array(divmod(state, self.size))

    def observation_to_int(self):
        return self.pos_to_state(self.agent_pos)

    def get_params_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(key)
        }

    def get_mapping_dict(self):
        return self.mapping_dict

    def load_from_dict(self, dict):
        self.__dict__.update(dict)
        self.rng = Generator(PCG64(seed=self.seed))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        if self.random_start:
            self.agent_pos = self.generate_pos()
        else:
            self.agent_pos = np.array(self.center_pos, dtype=np.float32)

        self.was_terminated = False

        return self.pos_to_state(self.agent_pos), {}

    def step(self, action):
        self.agent_pos = np.clip(
            self.agent_pos + self.action_to_direction[action], 0, self.size - 1
        )

        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0
        if reward == 1.0:
            if self.goal_only_once and self.was_terminated:
                reward = 0.0
            self.was_terminated = True
        terminated = True if reward and self.terminate_on_goal else False

        return self.pos_to_state(self.agent_pos), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full(
                (self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8
            )
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid

    def __eq__(self, other):
        if type(other) is type(self):
            is_equal = True
            is_equal = is_equal and (self.goal_pos == other.goal_pos)
            is_equal = is_equal and (self.seed == other.seed)

            return is_equal

        return False

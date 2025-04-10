import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary


class FlattenDictObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, Dict), "Only Dict observation space supported"
        self.obs_keys = list(env.observation_space.spaces.keys())

        # Flatten and concatenate all low/high from subspaces
        low = []
        high = []

        for key in self.obs_keys:
            space = env.observation_space.spaces[key]
            if isinstance(space, MultiBinary):
                low.append(np.zeros(space.n, dtype=np.float32))
                high.append(np.ones(space.n, dtype=np.float32))
            else:
                low.append(space.low.flatten())
                high.append(space.high.flatten())

        self.observation_space = Box(
            low=np.concatenate(low),
            high=np.concatenate(high),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = []
        for key in self.obs_keys:
            val = observation[key]
            val = val.astype(np.float32)
            obs.append(val.flatten())
        return np.concatenate(obs, axis=0)

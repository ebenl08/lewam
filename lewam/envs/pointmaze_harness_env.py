"""PointMaze harness, `lewam/PointMazeHarness-v0`: start and goal positions given per episode instead of from the dataset.

reset(options={"start_xy", "goal_xy"}) places the agent at start_xy and renders the goal frame with the agent at
goal_xy (OGBench's marker stays at (0, 0), as in every training frame). info["goal"] carries that frame; the
episode's last and smallest goal distance are kept for failure analysis.
"""

import gymnasium as gym
import numpy as np

from lewam.envs.pointmaze_env import PointMazeEnv

ENV_ID = "lewam/PointMazeHarness-v0"


class PointMazeHarnessEnv(PointMazeEnv):

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        if not options or "start_xy" not in options:
            return obs, info
        start_xy, goal_xy = np.asarray(options["start_xy"], np.float64), np.asarray(options["goal_xy"], np.float64)
        goal_state = np.concatenate([goal_xy, np.zeros(self.nq - 2 + self.nv)])
        self.set_state(goal_state)
        self._goal_img = self.render()
        self._set_goal_state(goal_state)
        self.set_state(np.concatenate([start_xy, np.zeros(self.nq - 2 + self.nv)]))
        self.last_distance = self.min_distance = self._goal_distance()
        return self._obs(), {"goal": self._goal_img, "goal_distance": self.last_distance}

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_distance = info["goal_distance"]
        self.min_distance = min(self.min_distance, self.last_distance)
        info["goal"] = self._goal_img
        return obs, reward, terminated, truncated, info


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=PointMazeHarnessEnv)

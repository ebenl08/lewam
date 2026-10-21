"""TwoRoom with the target frame as `info['goal']`, `lewam/TwoRoomHard-v1`, for manifest-mode evaluation.

Re-implements stable-worldmodel's `src/envs/tworoom/goal_wrapper.py` (the LeWAM env does not import PG code).
`TwoRoomEnv` renders the target frame at reset but does not return it; dataset-mode eval injects a recorded goal
frame instead, so only the episodic (manifest) path needs this wrapper.
"""

import gymnasium as gym
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

ENV_ID = "lewam/TwoRoomHard-v1"


class TwoRoomGoalWrapper(gym.Wrapper):
    """TwoRoomEnv with the rendered target frame as info['goal'] (HWC uint8, as env.render()).

    Also keeps the current episode's last and smallest agent-target distance (pixels), for failure analysis.
    """

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        info["goal"] = self._render_goal()
        self.last_distance = self.min_distance = float(info["distance_to_target"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["goal"] = self._render_goal()
        self.last_distance = float(info["distance_to_target"])
        self.min_distance = min(self.min_distance, self.last_distance)
        return obs, reward, terminated, truncated, info

    def _render_goal(self):
        return self.env.unwrapped._target_img.cpu().numpy().transpose(1, 2, 0)


def make_tworoom_hard(**kwargs):
    return TwoRoomGoalWrapper(TwoRoomEnv(**kwargs))


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=make_tworoom_hard)

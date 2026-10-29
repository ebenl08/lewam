"""Env adapters: task state and success-term errors, per env family."""

import gymnasium
import numpy as np


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """Angle wrapped to [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class EnvAdapter:
    """Task state of an env family and its success terms."""

    terms: tuple[str, ...] = ()     # names of the success terms, in the order errors() returns them

    def snapshot(self, env: gymnasium.Env) -> np.ndarray:
        """Task state of the live env."""
        raise NotImplementedError

    def goal(self, dataset, row: int) -> np.ndarray:
        """The same task state from a dataset row (h5py-style column access)."""
        raise NotImplementedError

    def errors(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Each success term divided by its tolerance; < 1 passes."""
        raise NotImplementedError


class PushTAdapter(EnvAdapter):
    """State (agent xy, block xy, block angle); terms 4-D position / 20, angle / 20 deg (PushT eval_state)."""

    terms = ("agent + block position", "block angle")

    def snapshot(self, env):
        return np.asarray(env.unwrapped._get_obs()[:5], dtype=np.float64)

    def goal(self, dataset, row):
        return np.asarray(dataset["state"][row][:5], dtype=np.float64)

    def errors(self, state, goal):
        position = np.linalg.norm(state[:4] - goal[:4]) / 20.0
        angle = abs(wrap_angle(state[4] - goal[4])) / (np.pi / 9)
        return np.array([position, angle])


class CubeAdapter(EnvAdapter):
    """State (EE xyz, EE yaw, cube xyz); AnyGoal criterion-3 terms."""

    terms = ("EE position", "EE yaw", "cube position")

    def snapshot(self, env):
        info = env.unwrapped.compute_ob_info()
        parts = [info["proprio/effector_pos"], info["proprio/effector_yaw"], info["privileged/block_0_pos"]]
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in parts])

    def goal(self, dataset, row):
        parts = [dataset["proprio_effector_pos"][row], dataset["proprio_effector_yaw"][row],
                 dataset["privileged_block_0_pos"][row]]
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in parts])

    def errors(self, state, goal):
        effector = np.linalg.norm(state[:3] - goal[:3]) / 0.04
        yaw = abs(wrap_angle(state[3] - goal[3])) / 0.10
        cube = np.linalg.norm(state[4:7] - goal[4:7]) / 0.04
        return np.array([effector, yaw, cube])


ADAPTERS = {
    "swm/PushT-v1": PushTAdapter,
    "anygoal/StackT-v2": PushTAdapter,
    "anygoal/AvoidT-v1": PushTAdapter,
    "anygoal/PushTAntigoal-v3": PushTAdapter,
    "swm/OGBCube-v0": CubeAdapter,
    "anygoal/CubeHop-v0": CubeAdapter,
    "anygoal/CubeDiagonal-v0": CubeAdapter,
}


class SnapshotLog(gymnasium.Wrapper):
    """Record adapter.snapshot after every step."""

    def __init__(self, env: gymnasium.Env, adapter: EnvAdapter):
        super().__init__(env)
        self.adapter = adapter
        self.snapshots = []

    def reset(self, **kwargs):
        self.snapshots = []
        return self.env.reset(**kwargs)

    def step(self, action):
        out = self.env.step(action)
        self.snapshots.append(self.adapter.snapshot(self.env))
        return out

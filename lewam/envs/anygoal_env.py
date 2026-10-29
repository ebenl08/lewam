"""AnyGoal's variant environments (`anygoal/AvoidT-v1`, `anygoal/StackT-v2`, `anygoal/PushTAntigoal-v3`,
`anygoal/TwoRoom-Route-v1`, `anygoal/CubeHop-v0`, `anygoal/CubeDiagonal-v0`), registered from the external checkout.

AnyGoal is not installed in the lewam env: its source is put on sys.path and its own `register_envs` registers the
environments with gym. It pins stable-worldmodel 0.1.1, the version the lewam env has.
"""

import sys
from pathlib import Path

ANYGOAL_ROOT = Path(__file__).resolve().parents[4] / "external" / "AnyGoal"
if str(ANYGOAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYGOAL_ROOT))

from anygoal.envs.registration import register_envs  # noqa: E402

register_envs()

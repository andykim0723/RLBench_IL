__version__ = '1.2.0'

import numpy as np
import pyrep

pr_v = np.array(pyrep.__version__.split('.'), dtype=int)
if pr_v.size < 4 or np.any(pr_v < np.array([4, 1, 0, 2])):
    raise ImportError(
        'PyRep version must be greater than 4.1.0.2. Please update PyRep.')


from sg_rlbench.environment import Environment
from sg_rlbench.action_modes.action_mode import ActionMode, ArmActionMode, GripperActionMode
from sg_rlbench.observation_config import ObservationConfig
from sg_rlbench.observation_config import CameraConfig
from sg_rlbench.sim2real.domain_randomization import RandomizeEvery
from sg_rlbench.sim2real.domain_randomization import VisualRandomizationConfig

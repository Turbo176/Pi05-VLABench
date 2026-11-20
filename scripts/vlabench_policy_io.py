"""Shared VLABench observation and actuator mapping for the pi0.5 pilot."""

import numpy as np

IMAGE_SIZE = 256
ACTUATOR_DIM = 9
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 50
SCENE_CAMERA = 'forward'
WRIST_CAMERA = 'franka/Franka_wrist_cam'


def _rgb(image):
    image = np.asarray(image)
    if image.shape == (3, IMAGE_SIZE, IMAGE_SIZE) and image.dtype == np.float32:
        # LeRobot decodes PNGs as CHW floats in [0, 1]. Recover the RGB bytes.
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError('Expected finite LeRobot RGB floats in [0, 1]')
        return np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError('Expected 256x256 HWC uint8 RGB or CHW float32 RGB')
    return image


def _policy_inputs(scene, wrist, state, prompt):
    scene, wrist = _rgb(scene), _rgb(wrist)
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (ACTUATOR_DIM,) or not np.isfinite(state).all():
        raise ValueError('Expected nine finite measured joint positions')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Expected the episode language instruction')
    return {
        'image': {
            'base_0_rgb': scene,
            'left_wrist_0_rgb': wrist,
            'right_wrist_0_rgb': np.zeros_like(scene),
        },
        'image_mask': {
            'base_0_rgb': np.True_,
            'left_wrist_0_rgb': np.True_,
            'right_wrist_0_rgb': np.False_,
        },
        'state': state,
        'prompt': prompt,
    }


def lerobot_inputs(row):
    """Map a LeRobot v2.1 row, with actions as training targets."""
    inputs = _policy_inputs(row['observation.images.forward'], row['observation.images.wrist'],
                            row['observation.state'], row['task'])
    actions = np.asarray(row['action'], dtype=np.float32)
    if actions.ndim not in (1, 2) or actions.shape[-1] != ACTUATOR_DIM or not np.isfinite(actions).all():
        raise ValueError('Expected finite nine-dimensional LeRobot action labels')
    return {**inputs, 'actions': actions}


def simulator_inputs(rgb_by_camera, state, instruction):
    """Map live observations by camera name with an explicit input allowlist."""
    return _policy_inputs(rgb_by_camera[SCENE_CAMERA], rgb_by_camera[WRIST_CAMERA], state, instruction)


def actuator_targets(actions):
    """Select the absolute joint targets from an unnormalized policy output."""
    actions = np.asarray(actions)
    if actions.ndim not in (1, 2) or actions.shape[-1] not in (ACTUATOR_DIM, MODEL_ACTION_DIM):
        raise ValueError('Expected a single command or chunk with 9 or 32 action coordinates')
    targets = actions[..., :ACTUATOR_DIM]
    if not np.issubdtype(targets.dtype, np.floating) or not np.isfinite(targets).all():
        raise ValueError('Actuator targets must be finite floats')
    return targets

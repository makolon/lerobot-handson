import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless GPU rendering on the GH200
# Miyabi sets CUDA_VISIBLE_DEVICES to a GPU *UUID* (e.g. "GPU-cd4ac836-..."), but
# robosuite's EGL device picker parses that var as an integer index and crashes. It
# prefers MUJOCO_EGL_DEVICE_ID when set, so pin it to 0 (each Miyabi-G node = 1 GPU).
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
from PIL import Image
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env

# build one LIBERO env from the libero_object suite
cfg = make_env(LiberoEnv(task="libero_object"), n_envs=1)   # {suite: {task_id: env}}
suite = next(iter(cfg))
env = cfg[suite][next(iter(cfg[suite]))]

obs, _ = env.reset(seed=0)
print("observation keys:", list(obs.keys()))

# find the first RGB image (HWC uint8) anywhere in the observation
def first_image(d):
    for v in d.values():
        if isinstance(v, dict):
            found = first_image(v)
            if found is not None:
                return found
        elif hasattr(v, "shape"):
            a = np.asarray(v)
            if a.ndim >= 3 and a.shape[-1] == 3 and a.dtype == np.uint8:
                return a.reshape(-1, a.shape[-3], a.shape[-2], 3)[0]
    return None

img = first_image(obs)
assert img is not None, "no RGB image in the observation — rendering likely failed"
Image.fromarray(img).save("libero_frame.png")
print("saved libero_frame.png", img.shape)
env.close()
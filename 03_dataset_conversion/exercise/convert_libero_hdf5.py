import argparse, os, h5py, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--hdf5", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--task", default=None)   # language; default: from file name
args = ap.parse_args()

f = h5py.File(args.hdf5, "r")
data = f["data"]

# language instruction: derive a readable one from the file name if not given
task = args.task
if task is None:
    base = os.path.basename(args.hdf5).replace("_demo.hdf5", "")
    task = base.replace("_", " ").strip()

fps = 20  # LIBERO / robosuite control runs at 20 Hz

# image size straight from the data (128 on most LIBERO versions, sometimes 256)
demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
H, W = data[demos[0]]["obs"]["agentview_rgb"].shape[1:3]

features = {
    "observation.images.image":  {"dtype": "video", "shape": (H, W, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (H, W, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action":            {"dtype": "float32", "shape": (7,), "names": ["action"]},
}

ds = LeRobotDataset.create(
    repo_id="local-user/libero-task", fps=fps, features=features,
    root=args.root, robot_type="panda", use_videos=True,
)

for name in demos:                       # demo_0, demo_1, ...
    g = data[name]
    obs = g["obs"]
    agent = obs["agentview_rgb"][()][:, ::-1]      # flip upside-down render
    wrist = obs["eye_in_hand_rgb"][()][:, ::-1]
    state = np.concatenate(
        [obs["ee_pos"][()], obs["ee_ori"][()], obs["gripper_states"][()]],
        axis=-1,
    ).astype("float32")                            # (T, 8)
    actions = g["actions"][()].astype("float32")   # (T, 7)
    for t in range(actions.shape[0]):
        ds.add_frame({
            "observation.images.image":  agent[t],
            "observation.images.image2": wrist[t],
            "observation.state": state[t],
            "action": actions[t],
            "task": task,                          # required every frame
        })
    ds.save_episode()

ds.finalize()
print("[convert] done:", args.root, "| episodes:", len(demos), "| task:", task)
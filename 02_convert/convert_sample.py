#!/usr/bin/env python
# =============================================================================
# convert_sample.py  —  合成データを LeRobotDataset (v3.0) 形式へ変換する最小例
# -----------------------------------------------------------------------------
# 目的: features/fps/robot_type を定義し、add_frame -> save_episode -> finalize
#       の一連を体験する。実機データ変換のテンプレートとして使う。
#
# 使い方:
#   python 02_convert/convert_sample.py          # ローカル保存のみ
#   python 02_convert/convert_sample.py --push   # HF_USER のリポジトリへ push
#
# TODO(lerobot): LeRobotDataset.create / add_frame / save_episode / finalize の
#                正確なシグネチャは v0.5.1 の docs / --help で要確認。
#                （v3.0 で finalize() が必須化された点は確認済み: PR #1903）
# =============================================================================
import argparse
import os

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_features(state_dim: int, action_dim: int, image_hw=(96, 96)):
    """各キーの dtype / shape / names を定義する。

    - observation.state, action : 低次元の連続値ベクトル（float32）
    - observation.images.front  : カメラ画像 (H, W, C) uint8 → 内部で動画エンコード
    """
    h, w = image_hw
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        "observation.images.front": {
            "dtype": "video",  # 画像列は video としてエンコードされる
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="HF_USER のリポジトリへ push する")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=7)
    args = parser.parse_args()

    # --- repo_id の決定（push 時は HF_USER 必須: fail-fast）---
    hf_user = os.environ.get("HF_USER", "")
    if args.push:
        if not hf_user or hf_user.startswith("<TODO"):
            raise SystemExit("ERROR: --push には HF_USER が必要です。`source config.env` してください。")
    repo_id = f"{hf_user or 'local-user'}/handson-convert-sample"

    image_hw = (96, 96)
    features = build_features(args.state_dim, args.action_dim, image_hw)

    # --- データセットを新規作成 ---
    # TODO(lerobot): create() の引数名（fps/features/robot_type/use_videos 等）を要確認。
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        robot_type="handson_dummy_arm",
        use_videos=True,
    )

    rng = np.random.default_rng(0)
    total_frames = 0
    for ep in range(args.episodes):
        for _ in range(args.frames_per_episode):
            frame = {
                "observation.state": torch.from_numpy(
                    rng.standard_normal(args.state_dim).astype(np.float32)
                ),
                "action": torch.from_numpy(
                    rng.standard_normal(args.action_dim).astype(np.float32)
                ),
                "observation.images.front": (
                    rng.integers(0, 256, size=(*image_hw, 3), dtype=np.uint8)
                ),
            }
            # TODO(lerobot): add_frame に task 文字列を渡す引数名を要確認
            #                （v0.5.1 では task をフレーム/エピソード単位で付与する）。
            dataset.add_frame(frame, task="pick up the cube")
            total_frames += 1
        dataset.save_episode()

    # --- 必ず finalize（呼ばないと parquet が壊れる）---
    dataset.finalize()
    print(f"[convert] {args.episodes} episodes / {total_frames} frames を書き出しました -> {repo_id}")

    if args.push:
        # ネット必要・ログインノードで実行すること
        dataset.push_to_hub()
        print(f"[convert] pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()

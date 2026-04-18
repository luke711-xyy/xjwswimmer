import json
from pathlib import Path
from typing import Optional

import cv2
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from Cluster_Env_ran import Cluster_Env_NRS_torch_C


# Set to a run directory name to override the latest run selection.
TARGET_RUN_NAME: Optional[str] = None

# Supported examples: "best_model", "final_model", "check_model_5000_steps"
TARGET_MODEL_NAME = "best_model"


def find_run_dir(runs_root: Path, target_run_name: Optional[str]) -> Path:
    if target_run_name:
        run_dir = runs_root / target_run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")
    return sorted(run_dirs, key=lambda path: path.name)[-1]


def main():
    script_dir = Path(__file__).resolve().parent
    runs_root = script_dir / "runs_cluster"
    run_dir = find_run_dir(runs_root, TARGET_RUN_NAME)

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)

    env_kwargs = run_config["env_kwargs"]
    model_path = run_dir / "models" / TARGET_MODEL_NAME
    video_name = f"Video_{run_dir.name}_{TARGET_MODEL_NAME}.mp4"

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model file not found: {model_path}.zip")

    env = DummyVecEnv([lambda: Cluster_Env_NRS_torch_C(**env_kwargs)])

    print(f"Loading run: {run_dir.name}")
    print(f"Loading model: {TARGET_MODEL_NAME}")
    model = SAC.load(str(model_path), env=env)

    max_frames = env_kwargs.get("max_episode_steps", 4000)
    print(f"Recording video to {video_name} (max_frames={max_frames})")

    obs = env.reset()
    frames = []

    for i in range(max_frames):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        frame = env.render()
        frames.append(frame)

        if dones[0]:
            info = infos[0] if infos else {}
            total_dx = info.get("episode_particle_total_dx", info.get("particle_total_dx", 0.0))
            print(f"Episode ended at frame {i + 1}; particle_total_dx={total_dx:.4f}")
            break

        if (i + 1) % 100 == 0:
            info = infos[0] if infos else {}
            print(
                "progress {frame}/{total} particle=({x:.4f}, {y:.4f}) dx={dx:.4f}".format(
                    frame=i + 1,
                    total=max_frames,
                    x=float(info.get("particle_x", 0.0)),
                    y=float(info.get("particle_y", 0.0)),
                    dx=float(info.get("particle_dx_step", 0.0)),
                )
            )

    if not frames:
        print("Video generation failed: no frames rendered.")
        return

    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print(f"Video generated: {video_name}")


if __name__ == "__main__":
    main()

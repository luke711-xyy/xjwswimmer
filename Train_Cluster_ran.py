import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from Cluster_Env_ran import Cluster_Env_NRS_torch_C


class RuntimeStatsCallback(BaseCallback):
    def __init__(self, position_print_every_n_steps: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.position_print_every_n_steps = position_print_every_n_steps

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        if not infos:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue

            self.logger.record_mean("env/particle_x", float(info.get("particle_x", 0.0)))
            self.logger.record_mean("env/particle_y", float(info.get("particle_y", 0.0)))
            self.logger.record_mean("env/particle_dx_step", float(info.get("particle_dx_step", 0.0)))
            self.logger.record_mean(
                "env/invalid_state_rate",
                1.0 if bool(info.get("invalid_state", False)) else 0.0,
            )
            self.logger.record_mean(
                "env/min_internal_angle_deg",
                float(info.get("min_internal_angle_deg", 180.0)),
            )

            invalid_reason = info.get("invalid_reason")
            self.logger.record_mean(
                "env/self_intersection_rate",
                1.0 if invalid_reason == "self_intersection" else 0.0,
            )
            self.logger.record_mean(
                "env/min_angle_violation_rate",
                1.0 if invalid_reason == "min_angle_violation" else 0.0,
            )

            if "episode_particle_total_dx" in info:
                self.logger.record_mean(
                    "env/particle_total_dx_episode",
                    float(info["episode_particle_total_dx"]),
                )

        if self.num_timesteps % self.position_print_every_n_steps == 0:
            first_info = infos[0]
            invalid_reason = first_info.get("invalid_reason") or "None"
            print(
                "step={step} particle=({x:.4f}, {y:.4f}) dx={dx:.4f} invalid={invalid}".format(
                    step=self.num_timesteps,
                    x=float(first_info.get("particle_x", 0.0)),
                    y=float(first_info.get("particle_y", 0.0)),
                    dx=float(first_info.get("particle_dx_step", 0.0)),
                    invalid=invalid_reason,
                )
            )

        return True


def build_monitored_env(env_kwargs: Dict[str, Any]):
    return Monitor(
        Cluster_Env_NRS_torch_C(**env_kwargs),
        info_keywords=("episode_particle_total_dx",),
    )


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).resolve().parent
    runs_root = script_dir / "runs_cluster"

    env_kwargs = {
        "control_period": 0.02,
        "link_num": 3,
        "num_robots": 3,
        "N_per_Seg": 3,
        "Q_per_Seg": 6,
        "epsilon": 0.02,
        "render_mode": "rgb_array",
        "max_episode_steps": 4000,
        "reward_scale": 2000.0,
        "invalid_penalty": -60.0,
        "min_internal_angle_deg": 60.0,
    }

    training_kwargs = {
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "device": "cpu",
        "train_freq": [10, "step"],
        "tau": 0.005,
        "gamma": 0.995,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "total_timesteps": 1_000_000,
        "eval_freq": 10_000,
        "n_eval_episodes": 5,
        "checkpoint_freq": 5_000,
    }
    monitoring_kwargs = {
        "position_print_every_n_steps": 10,
    }

    run_name = f"{timestamp}_sac_r{env_kwargs['num_robots']}_l{env_kwargs['link_num']}"
    run_dir = runs_root / run_name
    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_name": run_name,
        "timestamp": timestamp,
        "env_kwargs": env_kwargs,
        "training_kwargs": training_kwargs,
        "monitoring_kwargs": monitoring_kwargs,
        "reward_config": {
            "reward_mode": "particle_dx_signed",
            "reward_scale": env_kwargs["reward_scale"],
            "invalid_penalty": env_kwargs["invalid_penalty"],
        },
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    env = DummyVecEnv([lambda: build_monitored_env(env_kwargs)])
    eval_env = DummyVecEnv([lambda: build_monitored_env(env_kwargs)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=training_kwargs["eval_freq"],
        n_eval_episodes=training_kwargs["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=training_kwargs["checkpoint_freq"],
        save_path=str(model_dir),
        name_prefix="check_model",
    )
    runtime_callback = RuntimeStatsCallback(
        position_print_every_n_steps=monitoring_kwargs["position_print_every_n_steps"]
    )

    model = SAC(
        training_kwargs["policy"],
        env,
        verbose=1,
        device=training_kwargs["device"],
        train_freq=tuple(training_kwargs["train_freq"]),
        tau=training_kwargs["tau"],
        gamma=training_kwargs["gamma"],
        tensorboard_log=str(log_dir),
        learning_rate=training_kwargs["learning_rate"],
        batch_size=training_kwargs["batch_size"],
    )

    print(f"Training run directory: {run_dir}")
    print("-------------- Start training --------------")

    model.learn(
        total_timesteps=training_kwargs["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, runtime_callback],
        tb_log_name="sac",
    )

    model.save(str(model_dir / "final_model"))
    print(f"Training finished. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()

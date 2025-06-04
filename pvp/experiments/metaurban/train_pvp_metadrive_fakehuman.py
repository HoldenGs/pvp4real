# pvp/experiments/metaurban/train_pvp_metaurban_fakehuman.py

import argparse
import os
import uuid
from pathlib import Path

# === Pull in our MetaUrban FakeHumanEnv ===
from pvp.experiments.metaurban.egpo.fakehuman_env import FakeHumanEnv

# Everything else is identical, since PVPTD3, SB3 wrappers, etc. are framework‐agnostic
from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="pvp_metaurban_fakehuman", type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_team", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_human_proxy_value_loss", default="True", type=str)
    parser.add_argument("--with_agent_proxy_value_loss", default="True", type=str)
    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    args = parser.parse_args()

    # ——— Set up experiment directories ———
    experiment_batch_name = f"{args.exp_name}_freelevel{args.free_level}"
    seed = args.seed
    trial_name = f"{experiment_batch_name}_{get_time_str()}_{uuid.uuid4().hex[:8]}"
    print("Trial name is set to:", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] You are not logging to wandb right now!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / "runs" / experiment_batch_name
    os.makedirs(experiment_dir, exist_ok=True)
    trial_dir = experiment_dir / trial_name
    os.makedirs(trial_dir, exist_ok=False)
    print(f"Logging training data into {trial_dir}")

    free_level = args.free_level

    # ——— Build the config dictionary ———
    config = dict(
        # Env‐level config passed into our FakeHumanEnv.__init__()
        env_config=dict(
            use_render=True,           # show the MetaUrban window
            manual_control=False,      # remote actors won’t override
            free_level=free_level,     # probability threshold for takeover
            # (all other keys come from FakeHumanEnv.default_config())
        ),

        # Algorithm‐level config for PVPTD3 + HACO
        algo=dict(
            adaptive_batch_size=args.adaptive_batch_size,
            bc_loss_weight=args.bc_loss_weight,
            only_bc_loss=args.only_bc_loss,
            with_human_proxy_value_loss=args.with_human_proxy_value_loss,
            with_agent_proxy_value_loss=args.with_agent_proxy_value_loss,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # we train “reward‐free” with takeover cost only
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,                  # filled in below
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=str(trial_dir),
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir),
    )

    # ——— Instantiate and wrap the training environment ———
    train_env = FakeHumanEnv(config=config["env_config"])
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env

    # ——— (Optional) build an evaluation env if you want periodic evals ———
    # def _make_eval_env():
    #     eval_cfg = dict(use_render=False, manual_control=False, start_seed=1000, horizon=1500)
    #     from pvp.experiments.metaurban.human_in_the_loop_env import HumanInTheLoopEnv
    #     eval_env = HumanInTheLoopEnv(config=eval_cfg)
    #     eval_env = Monitor(env=eval_env, filename=str(trial_dir))
    #     return eval_env
    # eval_env = DummyVecEnv([_make_eval_env])
    eval_env = None

    # ——— Set up callbacks ———
    save_freq = args.save_freq
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq,
                           save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ——— Instantiate PVPTD3 ———
    model = PVPTD3(**config["algo"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading TD3+HACO checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ——— Begin training loop ———
    model.learn(
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )

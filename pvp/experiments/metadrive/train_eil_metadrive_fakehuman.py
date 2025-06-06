import argparse
import os
import uuid
from pathlib import Path

from pvp.experiments.metadrive.egpo.fakehuman_env import FakeHumanEnv
from pvp.eil import EIL
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.vec_env import SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="eil_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)
    parser.add_argument("--with_human_proxy_value_loss", default="True", type=str)
    parser.add_argument("--with_agent_proxy_value_loss", default="True", type=str)
    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = "{}_freelevel{}".format(args.exp_name, args.free_level)
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    free_level = args.free_level

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(

            # Original real human exp env config:
            # use_render=True,  # Open the interface
            # manual_control=True,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),

            # FakeHumanEnv config:
            free_level=free_level,
        ),

        # Algorithm config
        algo=dict(
            # intervention_start_stop_td=args.intervention_start_stop_td,
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
                discard_reward=True,  # We run in reward-free manner!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
            batch_size=args.batch_size,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    train_env = FakeHumanEnv(config=config["env_config"], )
    train_env = Monitor(env=train_env, filename=str(trial_dir))
    # Store all shared control data to the files.
    train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = SubprocVecEnv([_make_eval_env])

    # ===== Setup the callbacks =====
    save_freq = args.save_freq  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
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

    # ===== Setup the training algorithm =====
    model = EIL(**config["algo"])
    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=150,
        n_eval_episodes=1,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )

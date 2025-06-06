# pvp/experiments/metaurban/egpo/fakehuman_env.py

import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch

# === MetaUrban logging & expert imports ===
from metaurban.engine.logger import get_logger

# WE ASSUME you have created a MetaUrban expert checkpoint named "metaurban_pvp_20m_steps.zip"
from pvp.sb3.common.save_util import load_from_zip_file
from pvp.sb3.ppo import PPO
from pvp.sb3.ppo.policies import ActorCriticPolicy

from metaurban.policy.env_input_policy import EnvInputPolicy
from pvp.experiments.metaurban.human_in_the_loop_env import HumanInTheLoopEnv

FOLDER_PATH = pathlib.Path(__file__).parent
logger = get_logger()

# ——— Helper: load pretrained MetaUrban expert ———
def get_expert(env_instance):
    """
    Loads a PPO‐based expert policy trained on MetaUrban's HumanInTheLoopEnv.
    """
    # Make a fresh env so that the policy action shape matches MetaUrban's obs space.
    # train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        create_eval_env=False,
        verbose=2,
        device="auto",
        env=env_instance  # Use the existing environment instance
    )
    model = PPO(**algo_config)
    ckpt = FOLDER_PATH / "pretrained_policy_576k.zip"
    print(f"Loading MetaUrban PPO checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"MetaUrban expert loaded from {ckpt}!")
    # train_env.close()
    return model.policy


# Placeholder for the expert network
_expert = None


class FakeHumanEnv(HumanInTheLoopEnv):
    """
    Fake‐Human wrapper: On each step, we compute the expert's action distribution
    under MetaUrban's obs format, compare the agent's chosen action to the expert's,
    and "take over" if policy‐probability < (1 – free_level).
    """
    last_takeover = None
    last_obs = None
    expert = None

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("use_discrete", False):
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)
        
        # Load expert during initialization if not disabled
        if not self.config.get("disable_expert", False):
            try:
                global _expert
                if _expert is None:
                    _expert = get_expert(self)
                self.expert = _expert
                print("Expert successfully loaded during initialization!")
            except Exception as e:
                print(f"Warning: Failed to load expert during init: {e}")
                print("Expert will be disabled for this session.")
                self.config["disable_expert"] = True

    def setup_engine(self):
        """Override to set optimal text scale for better readability without clipping."""
        super().setup_engine()
        # Set optimal text scale for the on-screen display
        from metaurban.engine.core.onscreen_message import ScreenMessage
        ScreenMessage.SCALE = 0.05  # Smaller scale to prevent bottom clipping

    @property
    def action_space(self) -> gym.Space:
        if self.config.get("use_discrete", False):
            return gym.spaces.Discrete(self._num_bins**2)
        else:
            return super().action_space

    def default_config(self):
        cfg = super().default_config()
        cfg.update({
            "use_discrete": False,
            "disable_expert": False,
            "agent_policy": EnvInputPolicy,
            "free_level": 0.95,
            "manual_control": False,
            "use_render": False,
            "expert_deterministic": False,
            "agent_type": "wheelchair",  # Must match expert training - wheelchair not vehicle
        }, allow_add_new_key=True)
        return cfg

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        return self._actions[int(a)]

    def step(self, actions):
        """
        On each step:
         1. Convert discrete→continuous if needed.
         2. Query the expert network on self.last_obs to get distribution & expert_action.
         3. If expert_probability < (1 – free_level), force takeover: actions = expert_action.
         4. Call super().step(actions), record takeover flags/cost, etc.
        """
        actions = np.asarray(actions).astype(np.float32)

        if self.config.get("use_discrete", False):
            # Map index → continuous action
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover
        log_prob = None  # Initialize to avoid reference errors
        action_prob = None
        expert_action = None
        original_agent_action = copy.copy(actions)  # Store original action for display

        # — Expert takeover logic using last_obs —
        if not self.config.get("disable_expert", False) and self.expert is not None and self.last_obs is not None:

            # Convert the last observation to a tensor in the expert's device
            with torch.no_grad():
                obs_tensor, _ = self.expert.obs_to_tensor(self.last_obs)
                distribution = self.expert.get_distribution(obs_tensor)
                log_prob = distribution.log_prob(torch.from_numpy(actions).to(obs_tensor.device))
                action_prob = log_prob.exp().cpu().numpy()[0]  # Store for display

                # Expert action: sample or pick mode
                if self.config.get("expert_deterministic", False):
                    expert_action = distribution.mode().cpu().numpy()[0]
                else:
                    expert_action = distribution.sample().cpu().numpy()[0]

                # Decide takeover
                if action_prob < 1.0 - self.config["free_level"]:
                    if self.config.get("use_discrete", False):
                        # Force expert_action → discrete index → continuous
                        expert_idx = self.continuous_to_discrete(expert_action)
                        actions = self.discrete_to_continuous(expert_idx)
                    else:
                        actions = expert_action
                    self.takeover = True
                else:
                    self.takeover = False
        else:
            # If we disabled the expert or no observation available, never takeover
            self.takeover = False

        # Now call the base class's step (which will handle takeover cost, etc.)
        o, r, d, info = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if not self.config.get("disable_expert", False) and self.expert is not None and log_prob is not None:
            info["takeover_log_prob"] = log_prob.item()

        if self.config.get("use_discrete", False):
            info["raw_action"] = self.continuous_to_discrete(info["raw_action"])

        # Add visual rendering with comprehensive expert information
        if self.config.get("use_render", False):
            # Calculate takeover rate
            takeover_rate = np.mean(np.array(self.takeover_recorder) * 100) if len(self.takeover_recorder) > 0 else 0.0
            
            # Ultra-compact display to prevent bottom clipping
            text_dict = {
                # Essential expert info
                "Expert": "ON" if not self.config.get("disable_expert", False) and self.expert is not None else "OFF",
                "Free": f"{self.config.get('free_level', 0.95):.2f}",
                "Control": "EXPERT" if self.takeover else "AGENT",
                "Action": f"[{original_agent_action[0]:.2f}, {original_agent_action[1]:.2f}]",
            }
            
            # Add essential expert metrics if available
            if not self.config.get("disable_expert", False) and self.expert is not None:
                if action_prob is not None:
                    text_dict["Prob"] = f"{action_prob:.3f}"
                if expert_action is not None:
                    text_dict["Expert"] = f"[{expert_action[0]:.2f}, {expert_action[1]:.2f}]"
            
            # Essential statistics only
            text_dict.update({
                "Step": f"{self.total_steps}",
                "Takeover%": f"{takeover_rate:.1f}%",
                "Cost": f"{self.total_cost:.1f}",
                "Controls": "E=Pause",
            })
            
            # Render the text overlay
            super().render(text=text_dict)

        return o, r, d, info

    def _get_step_return(self, actions, engine_info):
        """
        We differ from the original only in that we do not call an expert policy
        inside _get_step_return; we already forced takeover above if needed.
        Everything else (cost stamping) happens as usual:
        """
        o, r, done_mask, cost_mask, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        # Update last_obs for the NEXT step
        self.last_obs = o
        d = done_mask or cost_mask
        last_t = self.last_takeover
        engine_info["takeover_start"] = (not last_t) and self.takeover
        engine_info["takeover"] = self.takeover

        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info.get("cost", 0)
        engine_info["episode_native_cost"] = self.episode_native_cost
        self.total_cost += engine_info.get("cost", 0)
        self.total_takeover_count += int(self.takeover)
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost

        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


if __name__ == "__main__":
    env = FakeHumanEnv(dict(free_level=0.95, use_render=False))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        if done:
            print(info)
            env.reset()

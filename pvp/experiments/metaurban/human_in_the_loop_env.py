# pvp/experiments/metaurban/human_in_the_loop_env.py

import copy
import time
from collections import deque

import numpy as np

# https://github.com/metadriverse/metaurban/blob/9f937640ce01a169dc2af46c7e7f8cc6c9bef00e/metaurban/engine/core/onscreen_message.py#L9
# https://github.com/metadriverse/metaurban/blob/9f937640ce01a169dc2af46c7e7f8cc6c9bef00e/metaurban/envs/sidewalk_static_env.py#L111
# https://github.com/metadriverse/metaurban/blob/9f937640ce01a169dc2af46c7e7f8cc6c9bef00e/metaurban/policy/manual_control_policy.py#L114
# Also includes TakeoverPolicyWithoutBrake, TODO: which to select?
from metaurban.engine.core.onscreen_message import ScreenMessage
from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.policy.manual_control_policy import TakeoverPolicyWithoutBrake
from metaurban.utils.math import safe_clip

ScreenMessage.SCALE = 0.1  # same as MetaDrive on-screen HUD scale

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    # MetaDrive used “traffic_density" but MetaUrban static env does NOT use traffic_density.:
    "object_density": 0.2,
    "crswalk_density": 1,  # Should only be for dynamic environment

    # Reward and cost setting:    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    # MetaUrban’s cost terms have identical names to MetaDrive’s Safe env (in MetaUrban’s code base,
    # they ship crash_vehicle_cost, out_of_route_cost, etc.). We’ll leave these as a basis to toggle later.
    "cost_to_reward": True,         # ADJUST: if MetaUrban’s static env uses same key
    "cos_similarity": False,        # If True, use cos–similarity takeover cost (same logic)

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyWithoutBrake,
    "controller": "keyboard",          # [keyboard, xbox, steering_wheel]
    "only_takeover_start_cost": False,

    # Visualize
    "vehicle_config": {
        "show_dest_mark": True,
        "show_line_to_dest": True,
        "show_line_to_navi_mark": True,
    },
    "horizon": 1500,
    
    # MetaUrban Static env will default to crash_vehicle_done=True, crash_object_done=True, etc.
    # To write a "safe mode" we override to keep episode alive on crash:
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "crash_building_done": False,
    "crash_human_done": False,
    # “out_of_route_done” already set above. If you want out_of_route to end episode, keep it True.
}

class HumanInTheLoopEnv(SidewalkStaticMetaUrbanEnv):
    """
    Human-in-the-loop Env Wrapper for the static sidewalk environment in MetaUrban.
    Add code for computing takeover cost and add information to the interface.
    """
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    takeover = False
    takeover_recorder = deque(maxlen=2000)
    agent_action = None
    in_pause = False
    start_time = time.time()

    def default_config(self):
        cfg = super().default_config()
        # Merge with HUMAN_IN_THE_LOOP_ENV_CONFIG, allowing new keys
        cfg.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return cfg

    def reset(self, *args, **kwargs):
        # Reset takeover flags before resetting the env
        self.takeover = False
        self.agent_action = None
        obs, info = super().reset(*args, **kwargs)
        # MetaUrban’s reset returns (obs, info) by default (Gymv26 style). We discard “info” here for legacy code.
        return obs

    def _get_step_return(self, actions, engine_info):
        """
        Called internally by MetaUrban after it computes (o, r, done_mask, cost_mask, engine_info).
        We override to (a) compute takeover cost, (b) stamp takeover flags into engine_info,
        (c) accumulate total cost / takeover counts.
        """
        # Step up the chain: MetaUrban’s static "_get_step_return" returns:
        # (obs, reward, termination_mask, cost_mask, engine_info)
        o, r, done_mask, cost_mask, engine_info = super()._get_step_return(actions, engine_info)
        # In MetaDrive you did “tm or tc”. Here, MetaUrban also packs termination_mask (True if done)
        d = done_mask or cost_mask

        # --- Figure out if the takeover just started this step ---
        shared_control_policy = self.engine.get_policy(self.agent.id)
        last_t = self.takeover
        self.takeover = getattr(shared_control_policy, "takeover", False)

        engine_info["takeover_start"] = (not last_t) and self.takeover
        engine_info["takeover"] = self.takeover

        # --- Decide whether to charge takeover_cost this step ---
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost

        # --- Stamp cumulative fields into info dict ---
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info.get("cost", 0)
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info.get("cost", 0)
        engine_info["total_cost"] = self.total_cost

        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        """
        In MetaDrive you overrode out_of_road to include sidewalk & on_lane tests.
        In MetaUrban, we simply trust MetaUrban's built-in _is_out_of_road. But if you need
        “crash_sidewalk” logic, you can reimplement here.
        """
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """
        Wrap the base step() so that:
         - we can record the raw agent_action (for takeover logic)
         - we can pause if user hits “e”
         - we stamp total_takeover_count and total_steps
         - we pop the HUD if use_render=True
        """
        self.agent_action = copy.copy(actions)
        ret = super().step(actions)

        # If we are paused by “e”, run the engine until unpaused
        while self.in_pause:
            self.engine.taskMgr.step()

        self.takeover_recorder.append(self.takeover)

        if self.config["use_render"]:
            super().render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        self.total_steps += 1
        self.total_takeover_count += int(self.takeover)
        ret[-1]["total_takeover_count"] = self.total_takeover_count
        return ret

    def stop(self):
        """Called when the user hits “e”; toggles pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Hook the “e” key to toggle pause on/off (exactly as MetaDrive)."""
        super().setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """
        If cos_similarity=False: always return cost=1.
        Otherwise: compute 1 - cosine_similarity(raw_action, agent_action).
        """
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action   = safe_clip(np.array(self.agent_action), -1, 1)
        mult = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        denom = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if denom < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = mult / denom
        return 1 - cos_dist


if __name__ == "__main__":
    # Same unit test as MetaDrive experiments
    env = HumanInTheLoopEnv({
        "manual_control": False,
        "use_render": False,
    })
    # env = HumanInTheLoopEnv({
    #     "manual_control": True,
    #     "use_render": True,
    # })
    env.reset()
    while True:
        x, _, done, _ = env.step([0, 0])
        if done:
            env.reset()

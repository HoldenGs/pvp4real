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
from metaurban.component.navigation_module.orca_navigation import ORCATrajectoryNavigation

ScreenMessage.SCALE = 0.06  # Further reduced to prevent bottom clipping

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    # MetaDrive used "traffic_density" but MetaUrban static env does NOT use traffic_density.:
    "object_density": 0.7,  # Increased from 0.2 to match MetaUrban example environment
    "crswalk_density": 1,  # Should only be for dynamic environment
    
    # Map and spawn configuration - IMPORTANT for proper object placement
    "map": "X",  # Use intersection maps like MetaUrban example (more challenging with objects)
    "random_spawn_lane_index": False,  # Consistent spawn positioning like MetaUrban example
    "walk_on_all_regions": False,  # Match MetaUrban example spawn behavior
    "accident_prob": 0,  # No accidents for cleaner training
    
    # Agent configuration - IMPORTANT: Must match expert training
    "agent_type": "wheelchair",  # Expert was trained for wheelchair control, not regular vehicle
    "drivable_area_extension": 55,  # From expert training config
    "height_scale": 1,  # From expert training config
    "spawn_deliveryrobot_num": 2,  # From expert training config
    "show_mid_block_map": False,  # From expert training config
    "show_ego_navigation": False,  # From expert training config
    "on_continuous_line_done": False,  # From expert training config
    "relax_out_of_road_done": True,  # From expert training config
    "max_lateral_dist": 15.0,  # From expert training config
    
    # Navigation setup for wheelchair agent - CRITICAL for proper destination setting
    "ego_navigation_module": ORCATrajectoryNavigation,  # Wheelchair uses EgoWheelchair which needs ego_navigation_module
    
    # Destination distance control - adjust to spawn closer to objects for more interactions
    "destination_range": (15, 50),  # Min and max destination distance in meters (default would be ~5-200m)
    
    # Spawn numbers from expert training (scaled for training efficiency)
    "spawn_human_num": 20,  # From expert training
    "spawn_wheelchairman_num": 1,  # From expert training  
    "spawn_edog_num": 2,  # From expert training
    "spawn_erobot_num": 1,  # From expert training
    "spawn_drobot_num": 1,  # From expert training
    "max_actor_num": 20,  # From expert training

    # Reward and cost setting:    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    # MetaUrban's cost terms have identical names to MetaDrive's Safe env (in MetaUrban's code base,
    # they ship crash_vehicle_cost, out_of_route_cost, etc.). We'll leave these as a basis to toggle later.
    "cost_to_reward": True,         # ADJUST: if MetaUrban's static env uses same key
    "cos_similarity": False,        # If True, use cos–similarity takeover cost (same logic)

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyWithoutBrake,
    "controller": "keyboard",          # [keyboard, xbox, steering_wheel]
    "only_takeover_start_cost": False,

    # Scene visualization and navigation
    "show_sidewalk": True,  # Make sidewalks visible
    "show_crosswalk": True,  # Make crosswalks visible 
    "vehicle_config": {
        "show_navi_mark": True,  # Show navigation markers like MetaUrban example
        "show_line_to_navi_mark": False,  # Match MetaUrban example
        "show_dest_mark": False,  # Match MetaUrban example
        "enable_reverse": True,  # Allow reverse like MetaUrban example
        "policy_reverse": False,  # Match MetaUrban example
    },
    "horizon": 1500,
    
    # MetaUrban Static env will default to crash_vehicle_done=True, crash_object_done=True, etc.
    # To write a "safe mode" we override to keep episode alive on crash:
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "crash_building_done": False,
    "crash_human_done": False,
    # "out_of_route_done" already set above. If you want out_of_route to end episode, keep it True.
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
    
    def __init__(self, config=None, *args, **kwargs):
        super(HumanInTheLoopEnv, self).__init__(config, *args, **kwargs)
        self.episode_native_cost = 0  # MetaUrban's _get_step_return does not provide this field so we compute it manually

    def default_config(self):
        cfg = super(HumanInTheLoopEnv, self).default_config()
        # Merge with HUMAN_IN_THE_LOOP_ENV_CONFIG, allowing new keys
        cfg.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        return cfg

    def reset(self, *args, **kwargs):
        # Reset takeover flags before resetting the env
        self.takeover = False
        self.agent_action = None
        # Reset manual accumualted cost
        self.episode_native_cost = 0
        # MetaUrban's reset returns (obs, info) by default (Gymv26 style). We discard "info" here for legacy code.
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        return obs

    def _get_step_return(self, actions, engine_info):
        """
        Called internally by MetaUrban after it computes (o, r, done_mask, cost_mask, engine_info).
        We override to (a) compute takeover cost, (b) stamp takeover flags into engine_info,
        (c) accumulate total cost / takeover counts.
        """
        # (obs, reward, termination_mask, cost_mask, engine_info)
        o, r, done_mask, cost_mask, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        d = done_mask or cost_mask  # termination_mask or cost_mask

        # Figure out if the takeover just started this step
        shared_control_policy = self.engine.get_policy(self.agent.id)
        last_t = self.takeover
        self.takeover = getattr(shared_control_policy, "takeover", False)
        engine_info["takeover_start"] = (not last_t) and self.takeover
        engine_info["takeover"] = self.takeover

        # Decide whether to charge takeover_cost this step
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost

        # Write cumulative fields into info dict
        native_cost = engine_info["cost"]
        
        # Episode local costs
        self.episode_native_cost += native_cost
        engine_info["native_cost"] = native_cost
        engine_info["episode_native_cost"] = self.episode_native_cost
        
        # Global costs
        self.total_cost += native_cost
        engine_info["total_cost"] = self.total_cost

        return o, r, done_mask, cost_mask, engine_info

    def _is_out_of_road(self, vehicle):
        """
        In MetaDrive you overrode out_of_road to include sidewalk & on_lane tests.
        In MetaUrban, we simply trust MetaUrban's built-in _is_out_of_road. But if you need
        "crash_sidewalk" logic, you can reimplement here.
        """
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """
        Wrap the base step() so that:
         - we can record the raw agent_action (for takeover logic)
         - we can pause if user hits "e"
         - we stamp total_takeover_count and total_steps
         - we pop the HUD if use_render=True
        """
        self.agent_action = copy.copy(actions)
        ret = super().step(actions)

        # Pause logic: if paused by "e", run the engine until unpaused
        while self.in_pause:
            self.engine.taskMgr.step()

        self.takeover_recorder.append(self.takeover)

        if self.config["use_render"]:
            # Calculate takeover rate
            takeover_rate = np.mean(np.array(self.takeover_recorder) * 100) if len(self.takeover_recorder) > 0 else 0.0
            
            # Ultra-compact text display to prevent bottom clipping
            text_dict = {
                # Essential info only
                "Status": "TAKEOVER" if self.takeover else "NORMAL",
                "Action": f"[{self.agent_action[0]:.2f}, {self.agent_action[1]:.2f}]" if self.agent_action is not None else "[0.00, 0.00]",
                "Step": f"{self.total_steps}",
                "Takeover%": f"{takeover_rate:.1f}%",
                "Cost": f"{round(self.total_cost, 1)}",
                "Controls": "E=Pause",
            }
            
            super().render(text=text_dict)

        self.total_steps += 1
        self.total_takeover_count += int(self.takeover)
        ret[-1]["total_takeover_count"] = self.total_takeover_count
        return ret

    def stop(self):
        """Called when the user hits "e"; toggles pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Hook the "e" key to toggle pause on/off (exactly as MetaDrive)."""
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
    # i = 0
    while True:
        obs, rewards, terminateds, truncateds, step_infos = env.step([0, 0])
        # print(f"step {i}: {step_infos}\n")
        # i += 1
        if terminateds:
            env.reset()

import time
import robosuite
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from robosuite.devices.keyboard import Keyboard
from robosuite.wrappers import VisualizationWrapper
from robosuite.controllers import load_composite_controller_config
from robosuite.environments import REGISTERED_ENVS

from tuli.envs import WipeSphere, WipeWashboard

# Register the environment
robosuite.environments.REGISTERED_ENVS["WipeSphere"] = WipeSphere
robosuite.environments.REGISTERED_ENVS["WipeWashboard"] = WipeWashboard

MAX_FR = 25  # max frame rate for running simluation

# BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
# controller_config = load_composite_controller_config(controller="BASIC")
controller_config = load_composite_controller_config(robot="Panda")
# breakpoint()

# # change later
# controller_config["body_parts"]["right"]["output_max"] = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5]
# controller_config["body_parts"]["right"]["output_min"] = [-0.1, -0.1, -0.1, -0.5, -0.5, -0.5]
# controller_config["body_parts"]["right"]["kp"] = 600

# create an environment for policy learning from low-dimensional observations
env = robosuite.make(
    "WipeSphere",
    robots=["Panda"],             # load a Sawyer robot and a Panda robot
    gripper_types="SphereGripper",                # use default grippers per robot arm
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    # horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
    ignore_done=True,                       # never terminate the environment
)

# Wrap this environment in a visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

pos_ensitivity = 1.0
rot_sensitivity = 1.0
device = Keyboard(env=env, pos_sensitivity=pos_ensitivity, rot_sensitivity=rot_sensitivity)
device.start_control()
env.robots[0].print_action_info_dict()
active_robot = env.robots[device.active_robot]

from copy import deepcopy

# breakpoint()
# success = env.perform_wiping_skill(num_cycles=1)
# if success:
#     print("Wiping skill completed successfully")
# else:
#     print("Wiping skill failed")
# breakpoint()

# do visualization
# for i in range(10000):
count = 0
while True: 
    start = time.time()
    # action = np.random.randn(*env.action_spec[0].shape)
    # breakpoint()

    input_ac_dict = device.input2action()
    action_dict = deepcopy(input_ac_dict)
    for arm in active_robot.arms:
        controller_input_type = active_robot.part_controllers[arm].input_type
        if controller_input_type == "delta":
            action_dict[arm] = input_ac_dict[f"{arm}_delta"]
        elif controller_input_type == "absolute":
            action_dict[arm] = input_ac_dict[f"{arm}_abs"]
        else:
            raise ValueError
    action = active_robot.create_action_vector(action_dict)
    print(f"action: {action}")

    obs, reward, done, _= env.step(action)
    # breakpoint()
    env.render()
    count += 1

    if count % 100 == 0:
        breakpoint()

    # limit frame rate if necessary
    elapsed = time.time() - start
    diff = 1 / MAX_FR - elapsed
    if diff > 0:
        time.sleep(diff)

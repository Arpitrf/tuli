import time
import robosuite
import numpy as np

from robosuite.controllers import load_composite_controller_config

MAX_FR = 25  # max frame rate for running simluation

# BASIC controller: arms controlled using OSC, mobile base (if present) using JOINT_VELOCITY, other parts controlled using JOINT_POSITION 
controller_config = load_composite_controller_config(controller="BASIC")

# create an environment for policy learning from low-dimensional observations
env = robosuite.make(
    "Wipe",
    robots=["Panda"],             # load a Sawyer robot and a Panda robot
    # gripper_types="default",                # use default grippers per robot arm
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
)

# do visualization
for i in range(10000):
    start = time.time()
    action = np.random.randn(*env.action_spec[0].shape)
    obs, reward, done, _ = env.step(action)
    env.render()

    # limit frame rate if necessary
    elapsed = time.time() - start
    diff = 1 / MAX_FR - elapsed
    if diff > 0:
        time.sleep(diff)

from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments import REGISTERED_ENVS
from robosuite.models.tasks import ManipulationTask
from .wipe_sphere_arena import WipeSphereArena
from .sphere_gripper import SphereGripper
import numpy as np
import matplotlib.pyplot as plt

class WipeSphere(Wipe):
    """
    A modified version of the Wipe environment that uses a table surface made of small spheres
    to provide vibration feedback during wiping.
    
    This class also supports temporally extended actions where a single action from the policy
    is executed for multiple environment steps.
    """
    
    def __init__(self, extended_action_steps=30, **kwargs):
        """
        Args:
            extended_action_steps (int): Number of environment steps to execute each action.
                                         Default is 30. Set to 1 to disable temporally extended actions.
            **kwargs: Additional arguments passed to parent Wipe class
        """
        # Override the gripper type to use our sphere gripper
        if "gripper_types" in kwargs:
            del kwargs["gripper_types"]
        
        # Store temporally extended action parameter
        self.extended_action_steps = extended_action_steps
        
        # Initialize parent class with our custom gripper
        super().__init__(gripper_types="SphereGripper", **kwargs)

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        # Load robots
        self._load_robots()
        
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        # Create custom arena
        mujoco_arena = WipeSphereArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
            sphere_radius=0.005,  # 5mm spheres
            sphere_spacing=0.015,  # 15mm spacing (0.015)
        )

        mujoco_arena._load_model()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Initialize objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    # TODO: Implement this method
    def _get_active_markers(self, c_geoms):
        return self.model.mujoco_arena.markers
    
    def step(self, action):
        """
        Override step to execute action for extended_action_steps.
        
        This method intercepts actions from the policy and executes them
        for multiple environment steps, accumulating rewards.
        
        Args:
            action (np.array): Action to execute
            
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment (after all steps)
                - (float) accumulated reward from all steps
                - (bool) whether the current episode is completed
                - (dict) misc information from the final step
        """
        if self.extended_action_steps <= 1:
            # If extended_action_steps is 1 or less, use normal step behavior
            return super().step(action)

        # Doing it here because if I do in reset, the variables are reset even before I can access them in the main script
        if self.timestep == 0:
            self.force_history = []
            self.all_peak_freqs = []
            self.contact_history = []
            self.rgb_image_list = []
        
        # Execute the action for multiple steps
        return self._execute_extended_action(action, self.extended_action_steps)
    
    def _execute_extended_action(self, action, num_steps):
        """
        Execute the same action for num_steps, accumulating rewards.
        
        Args:
            action (np.array): Action to execute repeatedly
            num_steps (int): Number of steps to execute the action
            
        Returns:
            4-tuple:
                - (OrderedDict) final observations after all steps
                - (float) accumulated reward (sum of all step rewards)
                - (bool) whether episode terminated
                - (dict) info from the final step
        """
        accumulated_reward = 0.0
        final_obs = None
        done = False
        final_info = {}

        # print("action: ", np.linalg.norm(action))
        
        # Execute the action for num_steps (or until episode ends)
        for step_idx in range(num_steps):
            # Check if episode is already done
            if self.done:
                # If already done, return the last observation we have
                if final_obs is None:
                    # Get current observation
                    final_obs = self._get_observations()
                break
            
            # Execute one step with the action
            obs, reward, done, info = super().step(action)

            # # save images
            # img = self.sim.render(height=256, width=256, camera_name="frontview")
            # img = img[::-1, :, :]
            img = np.zeros((256, 256, 3))
            self.rgb_image_list.append(img)

            # Accumulate reward
            accumulated_reward += reward
            
            # Store the observation (will be the final one if this is the last step)
            final_obs = obs
            final_info = info
            
            # If episode ended, break early
            if done:
                break
        
        # If we didn't get an observation (shouldn't happen, but safety check)
        if final_obs is None:
            final_obs = self._get_observations()
        
        # breakpoint()
        return final_obs, accumulated_reward, done, final_info

    def _check_success(self):
        return False

# Register the environment
REGISTERED_ENVS["WipeSphere"] = WipeSphere
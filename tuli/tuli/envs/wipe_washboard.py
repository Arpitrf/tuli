from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments import REGISTERED_ENVS
from robosuite.models.tasks import ManipulationTask
from .wipe_washboard_arena import WipeWashboardArena
from .sphere_gripper import SphereGripper
import numpy as np
import matplotlib.pyplot as plt

class WipeWashboard(Wipe):
    """
    A modified version of the Wipe environment that uses a washboard-like surface 
    made of half-cylinders to provide vibration feedback during wiping.
    """
    
    def __init__(self, **kwargs):
        # Override the gripper type to use our sphere gripper
        if "gripper_types" in kwargs:
            del kwargs["gripper_types"]
        
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
        mujoco_arena = WipeWashboardArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
            cylinder_radius=0.005,  # 5mm radius
            cylinder_spacing=0.02,  # 20mm spacing
        )

        mujoco_arena._load_model()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Initialize objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

# Register the environment
REGISTERED_ENVS["WipeWashboard"] = WipeWashboard

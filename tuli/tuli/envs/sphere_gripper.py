import numpy as np
import os
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.grippers import GRIPPER_MAPPING


class SphereGripper(GripperModel):
    """
    A simple spherical end-effector for wiping tasks
    """

    def __init__(self, idn=None, sphere_radius=0.02):  # 2cm radius by default
        # Store radius before calling parent init
        self.sphere_radius = sphere_radius
        
        # Get the path to our custom XML
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(curr_dir, "assets", "sphere_gripper.xml")
        
        # Initialize parent class with our custom XML
        super().__init__(xml_path, idn=idn)

        # Initialize gripper-specific parameters
        self._joints = []  # No joints to control
        self._actuators = []  # No actuators
        self._contact_geoms = ["sphere_collision", "sphere_contact"]
        self._visualization_geoms = ["sphere_visual"]

    def format_action(self, action):
        return np.array([])  # No joints to actuate

    @property
    def init_qpos(self):
        return np.array([])  # No joints to actuate

    @property
    def _important_geoms(self):
        """
        Returns:
            dict: Dictionary of important geom names mapped to their corresponding IDs
        """
        return {
            "sphere": [
                "sphere_collision",
                "sphere_visual",
            ],
            "corners": [
                "sphere_contact",  # This is the geom used for contact detection
            ],
        }

# Register the gripper
GRIPPER_MAPPING["SphereGripper"] = SphereGripper
import numpy as np
from robosuite.models.arenas import WipeArena
from robosuite.utils.mjcf_utils import new_joint, new_geom, new_body, new_site, array_to_string

class WipeWashboardArena(WipeArena):
    """A custom arena with a washboard-like surface made of half-cylinders for vibration feedback."""
    
    def __init__(
        self,
        table_full_size=(0.5, 0.8, 0.05),
        table_friction=(1.0, 0.1, 0.01),
        table_offset=(0.15, 0, 0.9),
        table_friction_std=0,
        coverage_factor=0.6,
        num_markers=100,
        line_width=0.04,
        two_clusters=False,
        cylinder_radius=0.005,  # 5mm radius
        cylinder_spacing=0.02,  # 20mm spacing between centers
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            table_friction_std=table_friction_std,
            coverage_factor=coverage_factor,
            num_markers=num_markers,
            line_width=line_width,
            two_clusters=two_clusters,
        )
        
        self.cylinder_radius = cylinder_radius
        self.cylinder_spacing = cylinder_spacing
        
    def _load_model(self):
        """Loads the arena model."""
        # super()._load_model()
        
        # Remove the default table collision and visual geoms
        table_body = self.worldbody.find("./body[@name='table']")
        table_body.remove(table_body.find("./geom[@name='table_collision']"))
        table_body.remove(table_body.find("./geom[@name='table_visual']"))
        
        # Add base table for stability
        base_visual = new_geom(
            name="table_base_visual",
            type="box",
            size=[self.table_full_size[0], self.table_full_size[1], self.table_full_size[2]/2],
            pos=[0, 0, -self.table_full_size[2]/2],
            rgba=[0.7, 0.7, 0.7, 1],
            group=1,  # visual group
            contype=0,
            conaffinity=0,
        )
        table_body.append(base_visual)

        # Calculate number of cylinders in x direction
        nx = int(self.table_full_size[0] * 2 / self.cylinder_spacing)
        
        # Create rows of half-cylinders
        for i in range(nx):
            x = -self.table_full_size[0] + i * self.cylinder_spacing
            
            # Visual cylinder
            cylinder_visual = new_geom(
                name=f"cylinder_visual_{i}",
                type="cylinder",
                size=[self.cylinder_radius, self.table_full_size[1]],  # radius and half-length
                pos=[x, 0, 0],
                quat=[0.707107, 0, 0.707107, 0],  # rotate 90 degrees around Y axis
                rgba=[0.9, 0.9, 0.9, 1],
                group=1,  # visual group
                contype=0,
                conaffinity=0,
            )
            table_body.append(cylinder_visual)
            
            # Collision cylinder
            collision_name = f"cylinder_collision_{i}"
            collision_geom = new_geom(
                name=collision_name,
                type="cylinder",
                size=[self.cylinder_radius * 1.05, self.table_full_size[1]],  # slightly larger
                pos=[x, 0, 0],
                quat=[0.707107, 0, 0.707107, 0],  # rotate 90 degrees around Y axis
                rgba=[0.5, 0.5, 0, 0.3],  # semi-transparent for debugging
                group=0,  # collision group
                contype=1,
                conaffinity=1,
                friction=array_to_string(self.table_friction),
            )
            table_body.append(collision_geom)

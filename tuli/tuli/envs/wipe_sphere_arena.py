import numpy as np
from robosuite.models.arenas import WipeArena
from robosuite.utils.mjcf_utils import new_joint, new_geom, new_body, new_site, array_to_string

class WipeSphereArena(WipeArena):
    """A custom arena with a table surface made of small spheres for vibration feedback."""
    
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
        sphere_radius=0.005,  # 5mm spheres
        sphere_spacing=0.015,  # 15mm spacing between sphere centers
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
        
        self.sphere_radius = sphere_radius
        self.sphere_spacing = sphere_spacing
        
    def _load_model(self):
        """Loads the arena model."""
        
        # Remove the default table collision and visual geoms
        table_body = self.worldbody.find("./body[@name='table']")
        table_body.remove(table_body.find("./geom[@name='table_collision']"))
        table_body.remove(table_body.find("./geom[@name='table_visual']"))
        
        # Calculate number of spheres in each dimension
        nx = int(self.table_full_size[0] * 2 / self.sphere_spacing)
        ny = int(self.table_full_size[1] * 2 / self.sphere_spacing)
        
        # Create grid of spheres
        for i in range(nx):
            for j in range(ny):
                x = -self.table_full_size[0] + i * self.sphere_spacing
                y = -self.table_full_size[1] + j * self.sphere_spacing
                z = 0  # At the table surface
                
                # Create sphere geom
                sphere_name = f"table_sphere_{i}_{j}"
                sphere_geom = new_geom(
                    name=sphere_name,
                    type="sphere",
                    size=[self.sphere_radius],
                    pos=[x, y, z],
                    rgba=[0.9, 0.9, 0.9, 1],
                    group=1,  # Visible group
                    friction=array_to_string(self.table_friction),
                )
                table_body.append(sphere_geom)
                
                # Create collision sphere (slightly larger for better contact)
                collision_name = f"table_sphere_collision_{i}_{j}"
                collision_geom = new_geom(
                    name=collision_name,
                    type="sphere",
                    size=[self.sphere_radius * 1.1],  # Slightly larger
                    pos=[x, y, z],
                    rgba=[0.5, 0.5, 0, 0.3],  # Semi-transparent for debugging
                    contype=1,  # Can collide with other objects
                    conaffinity=1,  # Other objects can collide with it
                    friction=array_to_string(self.table_friction),
                )
                table_body.append(collision_geom)
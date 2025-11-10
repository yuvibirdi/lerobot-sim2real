"""Gym wrappers for lighting and distractor object randomization in ManiSkill environments"""

import numpy as np
import gymnasium as gym
import sapien
from precompute_ycb_dimensions import filter_ycb_objects_by_dimensions

class LightingRandomizationWrapper(gym.Wrapper):
    """Randomizes lighting with shadows on each reset.

    Config params (from env_config.json):
        lighting_randomization: dict with keys:
            - enabled: bool (default: False)
            - ambient_range: [float, float] (default: [0.2, 0.6])
            - directional_color: [r, g, b] (default: [1.0, 1.0, 1.0])
            - num_lights: int (default: 1)
            - shadow_enabled: bool (default: True)
            - shadow_map_size: int (default: 4096)
    """

    def __init__(self, env, config: dict):
        super().__init__(env)
        self.config = config.get('lighting_randomization', {})
        self.enabled = self.config.get('enabled', False)

        if self.enabled:
            self.ambient_range = self.config.get('ambient_range', [0.2, 0.6])
            self.directional_color = self.config.get('directional_color', [1.0, 1.0, 1.0])
            self.num_lights = self.config.get('num_lights', 1)
            self.shadow_enabled = self.config.get('shadow_enabled', True)
            self.shadow_map_size = self.config.get('shadow_map_size', 4096)

            # Monkey-patch the environment's _load_lighting to randomize direction
            self._patch_load_lighting()

    def _patch_load_lighting(self):
        """Replace environment's _load_lighting to use random light directions per-environment"""
        original_load_lighting = self.unwrapped._load_lighting

        def randomized_load_lighting(options: dict):
            # Iterate over all sub-scenes (parallel environments) and set different lighting for each
            for sub_scene in self.unwrapped.scene.sub_scenes:
                # Set ambient light (can randomize this per the config if needed)
                if self.config.get('randomize_ambient', False):
                    ambient_intensity = np.random.uniform(self.ambient_range[0], self.ambient_range[1])
                    sub_scene.ambient_light = [ambient_intensity, ambient_intensity, ambient_intensity]
                else:
                    sub_scene.ambient_light = [0.3, 0.3, 0.3]

                # Add directional light(s) with RANDOM direction and shadows for each environment
                for _ in range(self.num_lights):
                    direction = np.random.uniform([-1, -1, -1], [1, 1, -0.5])
                    direction = direction / np.linalg.norm(direction)

                    sub_scene.add_directional_light(
                        direction,
                        self.directional_color,
                        shadow=self.shadow_enabled,
                        shadow_scale=10.0,
                        shadow_near=-10.0,
                        shadow_far=10.0,
                        shadow_map_size=self.shadow_map_size
                    )

        # Replace the method
        self.unwrapped._load_lighting = randomized_load_lighting

    def reset(self, **kwargs):
        # The lighting randomization happens in _load_lighting during reconfiguration
        # We need to force reconfiguration on every reset to get new light directions
        # Set reconfiguration_freq to 1 if not already set
        if self.enabled and self.unwrapped.reconfiguration_freq != 1:
            self.unwrapped.reconfiguration_freq = 1

        obs, info = super().reset(**kwargs)
        return obs, info


class DistractorObjectsWrapper(gym.Wrapper):
    """Adds random distractor objects to the scene on each reset.

    Config params (from env_config.json):
        distractor_objects: dict with keys:
            - enabled: bool (default: False)
            - num_range: [int, int] (default: [1, 3])
            - spawn_area_center: [x, y, z] (default: [0.0, 0.0, 0.02])
            - spawn_area_radius: float (default: 0.15)
            - size_range: [float, float] (default: [0.015, 0.025])
            - use_ycb: bool (default: False)
    """

    def __init__(self, env, config: dict):
        super().__init__(env)
        self.config = config.get('distractor_objects', {})
        self.enabled = self.config.get('enabled', False)

        if self.enabled:
            self.num_range = self.config.get('num_range', [1, 3])
            self.spawn_area_center = np.array(self.config.get('spawn_area_center', [0.0, 0.0, 0.02]))
            self.spawn_area_radius = self.config.get('spawn_area_radius', 0.15)
            self.size_range = self.config.get('size_range', [0.015, 0.025])
            self.use_ycb = self.config.get('use_ycb', False)

            # YCB model IDs (small objects)
            self.ycb_ids = filter_ycb_objects_by_dimensions(max_z=0.1)
            # Monkey-patch the environment's _load_scene to add distractors during reconfiguration
            self._patch_load_scene()

    def _patch_load_scene(self):
        """Replace environment's _load_scene to add distractors during reconfiguration"""
        original_load_scene = self.unwrapped._load_scene

        def randomized_load_scene(options: dict):
            # First call the original _load_scene to load all the normal objects
            original_load_scene(options)

            # Now add distractor objects to each sub-scene
            scene = self.unwrapped.scene
            num_distractors = np.random.randint(self.num_range[0], self.num_range[1] + 1)

            for i in range(num_distractors):
                # Random position in circular spawn area, slightly elevated to avoid floor collision
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.spawn_area_radius)
                x = self.spawn_area_center[0] + radius * np.cos(angle)
                y = self.spawn_area_center[1] + radius * np.sin(angle)
                z = self.spawn_area_center[2] + 0.05  # Elevate to avoid floor

                # Create actor builder
                if self.use_ycb:
                    try:
                        from mani_skill.utils.building import actors
                        model_id = np.random.choice(self.ycb_ids)
                        builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
                        builder.initial_pose = sapien.Pose(p=[x, y, z])
                    except Exception as e:
                        print(f"Failed to load YCB object: {e}, falling back to cube")
                        # Fallback to simple cube
                        builder = scene.create_actor_builder()
                        size = 0.02
                        builder.add_box_collision(half_size=[size, size, size])
                        builder.add_box_visual(half_size=[size, size, size], material=[0.8, 0.2, 0.2, 1.0])
                        builder.initial_pose = sapien.Pose(p=[x, y, z])
                else:
                    # Simple colored cube
                    builder = scene.create_actor_builder()
                    size = np.random.uniform(self.size_range[0], self.size_range[1])
                    builder.add_box_collision(half_size=[size, size, size])
                    builder.add_box_visual(
                        half_size=[size, size, size],
                        material=np.random.uniform(0, 1, 4)
                    )
                    builder.initial_pose = sapien.Pose(p=[x, y, z])

                # Build the distractor
                distractor = builder.build(name=f"distractor_{i}")

        # Replace the method
        self.unwrapped._load_scene = randomized_load_scene

    def reset(self, **kwargs):
        # Force reconfiguration on every reset to get new distractor positions/counts
        if self.enabled and self.unwrapped.reconfiguration_freq != 1:
            self.unwrapped.reconfiguration_freq = 1

        obs, info = super().reset(**kwargs)
        return obs, info

"""Gym wrappers for lighting and distractor object randomization in ManiSkill environments"""

import numpy as np
import gymnasium as gym
import sapien


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
        """Replace environment's _load_lighting to use random light directions"""
        original_load_lighting = self.unwrapped._load_lighting

        def randomized_load_lighting(options: dict):
            # Set constant ambient light (don't randomize)
            self.unwrapped.scene.set_ambient_light([0.3, 0.3, 0.3])

            # Add ONE directional light with RANDOM direction and shadows
            direction = np.random.uniform([-1, -1, -1], [1, 1, -0.5])
            direction = direction / np.linalg.norm(direction)

            self.unwrapped.scene.add_directional_light(
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

            self.distractors = []
            self.distractor_count = 0  # For unique naming across resets

            # YCB model IDs (small objects)
            self.ycb_ids = [
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "006_mustard_bottle",
                "009_gelatin_box",
                "010_potted_meat_can",
            ]

    def _add_distractors(self):
        """Spawn random distractor objects"""
        if not self.enabled:
            return

        scene = self.unwrapped.scene
        num_distractors = np.random.randint(self.num_range[0], self.num_range[1] + 1)

        # Track newly added distractors this reset
        new_distractors = []

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
                    # YCB objects have their own size, don't set initial pose rotation
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
                # Simple colored cube - not recommended, use YCB instead
                builder = scene.create_actor_builder()
                size = np.random.uniform(self.size_range[0], self.size_range[1])
                builder.add_box_collision(half_size=[size, size, size])
                builder.add_box_visual(
                    half_size=[size, size, size],
                    material=np.random.uniform(0, 1, 4)
                )
                builder.initial_pose = sapien.Pose(p=[x, y, z])

            # Build and track (use counter for unique names across resets)
            distractor = builder.build(name=f"distractor_{self.distractor_count}")
            new_distractors.append(distractor)
            self.distractor_count += 1

        # Replace old distractors list with new ones
        self.distractors = new_distractors

    def _remove_distractors(self):
        """Remove all distractor objects by moving them far away"""
        if not self.enabled or len(self.distractors) == 0:
            return

        # Move distractors far away from scene (GPU sim doesn't support remove_actor)
        for distractor in self.distractors:
            try:
                # Move far away and disable
                distractor.set_pose(sapien.Pose(p=[1000, 1000, 1000]))
            except Exception as e:
                # If that fails, try to hide it
                try:
                    # Some actors might have different APIs
                    if hasattr(distractor, 'hide'):
                        distractor.hide()
                except Exception:
                    pass

        # Don't clear the list - keep references so names stay unique
        # But mark them as "removed" so we know they're not active

    def reset(self, **kwargs):
        self._remove_distractors()
        obs, info = super().reset(**kwargs)
        self._add_distractors()
        return obs, info

    def close(self):
        self._remove_distractors()
        super().close()

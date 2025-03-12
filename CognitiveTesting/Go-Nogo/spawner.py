import pygame
import random
import colorsys
from go_no_settings import SCREEN_HEIGHT, SCREEN_WIDTH, TASK_TIME

def apply_uniform_hue(image_surface):
    # Choose a random hue shift
    hue_shift = random.uniform(0, 1)

    # Get image dimensions
    width, height = image_surface.get_size()

    # Access the pixel data of the surface
    pixels = pygame.PixelArray(image_surface)

    for x in range(width):
        for y in range(height):
            # Get the current pixel color in RGB
            r, g, b, a = image_surface.unmap_rgb(pixels[x, y])

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

            # Apply the same hue shift to all pixels
            h = (h + hue_shift) % 1.0

            # Convert HSV back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            # Convert back to 0-255 range and set the pixel
            pixels[x, y] = pygame.Color(int(r * 255), int(g * 255), int(b * 255), a)

    # Remove pixel array lock
    del pixels

    return image_surface

class Spawner(pygame.sprite.Sprite):
    def __init__(self, spawn_point, image_path, target, shuriken_type, speed=7, scale=(125, 125), change_color=False):
        super().__init__()
        original_image = pygame.image.load(image_path).convert_alpha()
        original_width, original_height = original_image.get_size()
        TARGET_HEIGHT = SCREEN_HEIGHT // 9  # Adjust height based on screen size
        aspect_ratio = original_width / original_height
        TARGET_WIDTH = int(TARGET_HEIGHT * aspect_ratio)  # Maintain aspect ratio
        self.image = pygame.transform.scale(original_image, (TARGET_WIDTH, TARGET_HEIGHT))

        if change_color:
            self.image = apply_uniform_hue(self.image)
        self.original_image = self.image 
        self.rect = self.image.get_rect(center=spawn_point)
        self.target = pygame.math.Vector2(target)  # Center of the hexagon
        self.position = pygame.math.Vector2(spawn_point)  # Initial position
        self.velocity = (self.target - self.position).normalize() * speed  # Calculate direction
        self.type = shuriken_type
        self.destroying = False
        self.scale_factor = 0.9  # For shrinking animation
        self.spawn_time = pygame.time.get_ticks()  # Record spawn time
        self.stationary_duration = TASK_TIME  # Stay in place for 2 seconds before moving
        self.flying = False  # Track if the shuriken is in the flight phase
    
    def slash(self):
        """Initiate the destroying (shrinking) effect."""
        if not self.flying:  # Prevent slashing during the flight phase
            self.destroying = True
    
    def update(self):
        current_time = pygame.time.get_ticks()
        if not self.flying:
            if self.destroying:
                # Shrink the shuriken gradually until it disappears
                new_width = max(1, int(self.rect.width * self.scale_factor))
                new_height = max(1, int(self.rect.height * self.scale_factor))

                # Apply the scaled image
                self.image = pygame.transform.scale(self.original_image, (new_width, new_height))
                self.rect = self.image.get_rect(center=self.rect.center)

                # Remove the shuriken once it's sufficiently small
                if new_width <= 1 or new_height <= 1:
                    self.kill()
                return 

        # Only start flying after the stationary period if not destroying.
        if not self.flying and (current_time - self.spawn_time >= self.stationary_duration):
            self.flying = True

        if self.flying:
            # Move toward the target after the stationary phase.
            self.position += self.velocity
            self.rect.center = self.position

            # If the shuriken reaches near the target, remove it.
            if self.position.distance_to(self.target) < 5:
                self.kill()

def spawn_shuriken(shuriken_group, hexagon_points, hexagon_center, change_color=False, is_inhabitation=True, is_vigilance=False):
    # Select a random spawn point from the hexagon vertices
    spawn_point = random.choice(hexagon_points)
    if spawn_point == hexagon_points[0]: prompt = "right"
    elif spawn_point == hexagon_points[1]: prompt = "top_right"
    elif spawn_point == hexagon_points[2]: prompt = "top_left"
    elif spawn_point == hexagon_points[3]: prompt = "left"
    elif spawn_point == hexagon_points[4]: prompt = "bottom_left"
    elif spawn_point == hexagon_points[5]: prompt = "bottom_right"
    # Weighted random choice to set either "go" or "dontgo" type
    if is_inhabitation:
        if random.random() < 0.8:
            shuriken_image = 'CognitiveTesting/Go-Nogo/assets/sprites/shuriken.png'
            shuriken_type = "Go"
        else:
            shuriken_image = 'CognitiveTesting/Go-Nogo/assets/sprites/heart.png'
            shuriken_type = "Dontgo"
    elif is_vigilance:
        if random.random() < 0.2:
            shuriken_image = 'CognitiveTesting/Go-Nogo/assets/sprites/shuriken.png'
            shuriken_type = "Go"
        else:
            shuriken_image = 'CognitiveTesting/Go-Nogo/assets/sprites/heart.png'
            shuriken_type = "Dontgo"
    
    # Create the shuriken and add it to the group
    shuriken = Spawner(
        spawn_point=spawn_point,
        image_path=shuriken_image,
        target=hexagon_center,
        shuriken_type=shuriken_type,
        speed=7,
        change_color=change_color
    )
    shuriken_group.add(shuriken)
    return prompt

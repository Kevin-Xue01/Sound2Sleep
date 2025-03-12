import pygame
from go_no_settings import SCREEN_HEIGHT
import os

class Player(pygame.sprite.Sprite):
    def __init__(self, idle_folder, bottom_left, bottom_right, top_left, top_right, left, right, position):
        super().__init__()
        # Load frames with different scaling factors
        self.idle_frames = self.load_images(idle_folder, scale_factor=9)  # Default scaling
        
        # Slightly larger for top and bottom movements
        self.bottom_left_frames = self.load_images(bottom_left, scale_factor=6)
        self.bottom_right_frames = self.load_images(bottom_right, scale_factor=6)
        self.top_left_frames = self.load_images(top_left, scale_factor=4)
        self.top_right_frames = self.load_images(top_right, scale_factor=4)

        # Standard size for left and right
        self.left_frames = self.load_images(left, scale_factor=9)
        self.right_frames = self.load_images(right, scale_factor=9)

        self.image = self.idle_frames[0]  
        self.rect = self.image.get_rect(center=position)
        self.animation_index = 0
        self.slashing = False
        self.can_slash = True
        self.current_frames = self.idle_frames 

    def load_images(self, folder_path, scale_factor=9):
        """Load images while maintaining aspect ratio and applying different scaling factors."""
        frames = []
        for filename in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, filename)
            img = pygame.image.load(img_path).convert_alpha()

            # Get original dimensions
            original_width, original_height = img.get_size()

            # Scale height first
            target_height = SCREEN_HEIGHT // scale_factor

            # Maintain aspect ratio
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)

            # Resize the image
            img = pygame.transform.scale(img, (target_width, target_height))
            frames.append(img)

        return frames

    def update(self, prompt):
        """Update the player's animation based on the current state."""
        # Set appropriate animation frames based on the state
        if self.slashing:
            # Get the correct frames for slashing based on the prompt
            self.current_frames = self.get_slash_frames(prompt)
        else:
            # Default to idle frames when not slashing
            self.current_frames = self.idle_frames

        # Adjust position for specific prompts
        base_y = SCREEN_HEIGHT // 2 - self.rect.height // 2
        self.rect.y = base_y
        if prompt in ["top_left", "top_right"] and self.slashing:
            self.rect.y -= SCREEN_HEIGHT // 32  # Move up when slashing top prompts

        # Update animation index
        self.animation_index += 0.1
        if self.animation_index >= len(self.current_frames):
            self.animation_index = 0
            if self.slashing:
                self.slashing = False  # End slashing animation

        # Set the current frame
        self.image = self.current_frames[int(self.animation_index)]

    def get_slash_frames(self, prompt):
        """Return the appropriate slash frames based on the prompt."""
        slash_map = {
            'bottom_left': self.bottom_left_frames,
            'bottom_right': self.bottom_right_frames,
            'top_left': self.top_left_frames,
            'top_right': self.top_right_frames,
            'left': self.left_frames,
            'right': self.right_frames
        }
        return slash_map.get(prompt, self.idle_frames)  # Default to idle frames

    def slash(self):
        """Trigger the slash animation if slashing is allowed."""
        if self.can_slash and not self.slashing:
            self.slashing = True
            self.animation_index = 0
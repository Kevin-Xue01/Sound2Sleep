import pygame
import sys
import os
import random

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Load assets
background = pygame.image.load('assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
platform_img = pygame.image.load('assets/vpat/platform.png').convert_alpha()
idle_img = pygame.image.load('assets/sprites/player_idle/1.png').convert_alpha()
floor_img = pygame.image.load('assets/vpat/floor.png').convert_alpha()
floor_img = pygame.transform.scale(floor_img, (150, 50))  # Adjust width and height as needed

shuriken_img = pygame.image.load('assets/sprites/shuriken.png').convert_alpha()
shuriken_img = pygame.transform.scale(shuriken_img, (50, 50))

shuriken_active = False
shuriken_rect = shuriken_img.get_rect(center=(SCREEN_WIDTH // 2 - 300, SCREEN_HEIGHT // 2))
shuriken_target_reached = False
shuriken_shrinking = False

def handle_shuriken():
    """Handle the shuriken's movement, shrinking, and interactions."""
    global shuriken_active, shuriken_target_reached, shuriken_shrinking, hurt_animation_active, hurt_frame_index, hurt_timer, powerup_animation_active, powerup_frame_index, powerup_timer

    if shuriken_active:
        if not shuriken_target_reached:
            # Move shuriken toward the player if the guess is incorrect
            shuriken_rect.x += 10
            if shuriken_rect.colliderect(player_rect):
                shuriken_target_reached = True
                hurt_animation_active = True
                hurt_frame_index = 0
                hurt_timer = pygame.time.get_ticks()
        elif shuriken_shrinking:
            if not swipe_animation_active:
                swipe_animation_active = True
                swipe_frame_index = 0
                swipe_timer = pygame.time.get_ticks()
            else:
                # Shrink the shuriken size and position proportionally
                shuriken_rect.width -= 2
                shuriken_rect.height -= 2
                shuriken_rect.x += 1  # Adjust position to keep it centered
                shuriken_rect.y += 1

                if shuriken_rect.width <= 0:
                    shuriken_active = False
                    shuriken_shrinking = False
                    powerup_animation_active = True
                    powerup_frame_index = 0
                    powerup_timer = pygame.time.get_ticks()
                    feedback_message = ""  # Clear feedback after animation
                    pair_index += 1  # Move to the next pair
                    if pair_index >= len(pairs):  # Check if all pairs are done
                        pairs_shown = True


# Load animations
hurt_frames = [pygame.image.load(f'assets/sprites/hurt/{i}.png').convert_alpha() for i in range(1, 4)]
powerup_frames = [pygame.image.load(f'assets/sprites/powerup/{i}.png').convert_alpha() for i in range(1, 4)]
swipe_frames = [pygame.image.load(f'assets/sprites/player_swipe/{i}.png').convert_alpha() for i in range(1, 4)]

hurt_frames = [pygame.transform.scale(frame, (80, 70)) for frame in hurt_frames]
powerup_frames = [pygame.transform.scale(frame, (80, 70)) for frame in powerup_frames]
swipe_frames = [pygame.transform.scale(frame, (80, 70)) for frame in swipe_frames]

# Scale platforms and idle
platform_img = pygame.transform.scale(platform_img, (150, 150))
idle_img = pygame.transform.scale(idle_img, (80, 70))

# Define positions
FLOOR_Y = SCREEN_HEIGHT // 2 - 100  # Position slightly below platforms
floor_rect = floor_img.get_rect(midtop=(SCREEN_WIDTH // 2, FLOOR_Y))
PLATFORM_Y = SCREEN_HEIGHT // 2 + 230
PLATFORM_SPACING = 150  # Distance between platforms
platform1_pos = (SCREEN_WIDTH // 2 - PLATFORM_SPACING, PLATFORM_Y)
platform2_pos = (SCREEN_WIDTH // 2 + PLATFORM_SPACING, PLATFORM_Y)
player_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100)

# Load all icons
icon_path = 'assets/vpat/icons'
icons = [pygame.image.load(os.path.join(icon_path, file)).convert_alpha() for file in os.listdir(icon_path)]
icons = [pygame.transform.scale(icon, (70, 70)) for icon in icons]

# Game variables
icon_history = []
pairs = []
new_pairs = []
score = {"Correct": 0, "Incorrect": 0}
pair_index = 0
pairs_shown = False
round_complete = False
show_transition_message = False
start_message_shown = True
start_message_start_time = pygame.time.get_ticks()
START_MESSAGE_DURATION = 2000  # 2 seconds for "Remember These Pairs"
transition_start_time = None
TRANSITION_DURATION = 3000  # 3 seconds for "Now Time to Remember!"
# Generate 10 unique pairs
random.shuffle(icons)
pairs = [(icons[i], icons[i + 1]) for i in range(0, 20, 2)]
icon_history.extend(pairs)

# Animation settings
hurt_animation_active = False
hurt_frame_index = 0
hurt_timer = 0
HURT_INTERVAL = 150

powerup_animation_active = False
powerup_frame_index = 0
powerup_timer = 0
POWERUP_INTERVAL = 150

swipe_animation_active = False
swipe_frame_index = 0
swipe_timer = 0
SWIPE_INTERVAL = 150

def display_pair(pair):
    """Display a pair of icons centered on the platforms."""
    left_icon, right_icon = pair
    # Center the icons on the platforms
    left_rect = left_icon.get_rect(center=platform1_rect.center)
    right_rect = right_icon.get_rect(center=platform2_rect.center)
    screen.blit(left_icon, left_rect)
    screen.blit(right_icon, right_rect)

# Feedback settings
feedback_message = ""
feedback_start_time = None
FEEDBACK_DURATION = 1000  # 1 second for feedback display

def display_feedback():
    """Display feedback text on the screen."""
    if feedback_message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(feedback_message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 180)))
        
# Game loop
running = True
while running:
    screen.blit(background, (0, 0))

    # Draw platforms
    platform1_rect = platform_img.get_rect(midbottom=platform1_pos)
    platform2_rect = platform_img.get_rect(midbottom=platform2_pos)
    screen.blit(platform_img, platform1_rect)
    screen.blit(platform_img, platform2_rect)

    # Draw floor
    screen.blit(floor_img, floor_rect)

    # Handle animations and idle player
    if hurt_animation_active:
        if pygame.time.get_ticks() - hurt_timer > HURT_INTERVAL:
            hurt_frame_index += 1
            hurt_timer = pygame.time.get_ticks()
            if hurt_frame_index >= len(hurt_frames):
                hurt_animation_active = False
                hurt_frame_index = 0
                feedback_message = ""  # Clear feedback after hurt animation
        else:
            screen.blit(hurt_frames[hurt_frame_index], player_rect)
    elif powerup_animation_active:
        if pygame.time.get_ticks() - powerup_timer > POWERUP_INTERVAL:
            powerup_frame_index += 1
            powerup_timer = pygame.time.get_ticks()
            if powerup_frame_index >= len(powerup_frames):
                powerup_animation_active = False
                powerup_frame_index = 0
                feedback_message = ""  # Clear feedback after power-up animation
        else:
            screen.blit(powerup_frames[powerup_frame_index], player_rect)
    elif swipe_animation_active:
        if pygame.time.get_ticks() - swipe_timer > SWIPE_INTERVAL:
            swipe_frame_index += 1
            swipe_timer = pygame.time.get_ticks()
            if swipe_frame_index >= len(swipe_frames):
                swipe_animation_active = False
                swipe_frame_index = 0
        else:
            screen.blit(swipe_frames[swipe_frame_index], player_rect)
    else:
        # Draw idle player only if no animations are active
        player_rect = idle_img.get_rect(center=player_pos)
        screen.blit(idle_img, player_rect)

    # Handle shuriken
    if shuriken_active:
        if shuriken_target_reached:
            # Move shuriken toward the player on incorrect choice
            shuriken_rect.x += 10  # Adjust speed here
            screen.blit(shuriken_img, shuriken_rect)
            if shuriken_rect.colliderect(player_rect):
                shuriken_active = False
                hurt_animation_active = True
                hurt_frame_index = 0
                hurt_timer = pygame.time.get_ticks()
                feedback_message = ""  # Clear the feedback after animation
        elif shuriken_shrinking:
            # Play swipe animation first
            if not swipe_animation_active:
                swipe_animation_active = True
                swipe_frame_index = 0
                swipe_timer = pygame.time.get_ticks()
            else:
                # Shrink the shuriken after swipe animation starts
                shuriken_rect.width -= 2
                shuriken_rect.height -= 2
                shuriken_rect.x += 1  # Adjust position to keep it centered
                shuriken_rect.y += 1
                screen.blit(shuriken_img, shuriken_rect)
                if shuriken_rect.width <= 0:
                    shuriken_active = False
                    shuriken_shrinking = False
                    powerup_animation_active = True
                    powerup_frame_index = 0
                    powerup_timer = pygame.time.get_ticks()
                    feedback_message = ""  # Clear feedback after animation
                    pair_index += 1  # Move to the next pair
                    if pair_index >= len(pairs):  # Check if all pairs are done
                        pairs_shown = True
        else:
            # Display stationary shuriken
            screen.blit(shuriken_img, shuriken_rect)

    # Show "Remember These Pairs" at the start
    if start_message_shown:
        font = pygame.font.SysFont(None, 50)
        text = font.render("Remember These Pairs", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        # Check if the start message duration has passed
        if pygame.time.get_ticks() - start_message_start_time > START_MESSAGE_DURATION:
            start_message_shown = False
    else:
        # Display feedback
        if feedback_message:
            font = pygame.font.SysFont(None, 50)
            text = font.render(feedback_message, True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 180)))

        # Show current pair or transition message
        if not pairs_shown:
            if feedback_message and pygame.time.get_ticks() - feedback_start_time > FEEDBACK_DURATION:
                feedback_message = ""  # Clear feedback before the next pair
                pair_index += 1
                if pair_index == len(pairs):
                    pairs_shown = True
                    show_transition_message = True
                    transition_start_time = pygame.time.get_ticks()
            elif pair_index < len(pairs):
                display_pair(pairs[pair_index])
        elif show_transition_message:
            # Show "Now Time to Remember!" message
            font = pygame.font.SysFont(None, 50)
            text = font.render("Now Time to Remember!", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

            # Check if the transition period has ended
            if pygame.time.get_ticks() - transition_start_time > TRANSITION_DURATION:
                show_transition_message = False
                pair_index = 0  # Reset for new pairs
                new_pairs = icon_history[:5] + pairs[:5]
                random.shuffle(new_pairs)
        elif not round_complete:
            if feedback_message and pygame.time.get_ticks() - feedback_start_time > FEEDBACK_DURATION:
                feedback_message = ""  # Clear feedback before the next pair
                pair_index += 1
                if pair_index == len(new_pairs):
                    round_complete = True
            else:
                if not shuriken_active:
                    shuriken_active = True
                    shuriken_rect = shuriken_img.get_rect(center=(SCREEN_WIDTH // 2 - 300, player_pos[1] + 5))
                    shuriken_target_reached = False
                display_pair(new_pairs[pair_index])
        else:
            # All pairs shown
            font = pygame.font.SysFont(None, 50)
            text = font.render("All pairs shown!", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            # Allow spacebar to move to the next pair only during the "Remember These Pairs" phase
            if start_message_shown or (not pairs_shown and feedback_message == ""):
                if event.key == pygame.K_SPACE and not pairs_shown:
                    pair_index += 1
                    if pair_index == len(pairs):
                        pairs_shown = True
                        show_transition_message = True
                        transition_start_time = pygame.time.get_ticks()

            # Identify pairs after initial "Remember These Pairs" phase
            elif event.key in [pygame.K_LEFT, pygame.K_RIGHT] and not show_transition_message and not feedback_message:
                current_pair = new_pairs[pair_index]
                is_reused = current_pair in icon_history
                user_choice = event.key == pygame.K_RIGHT  # Right for reused, Left for new

                if user_choice == is_reused:
                    score["Correct"] += 1
                    feedback_message = "Correct"
                    shuriken_shrinking = True
                else:
                    score["Incorrect"] += 1
                    feedback_message = "Incorrect"
                    shuriken_target_reached = True

                feedback_start_time = pygame.time.get_ticks()

    # Update screen and FPS
    pygame.display.flip()
    clock.tick(FPS)

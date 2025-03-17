import pygame
import sys
from spawner import spawn_shuriken
from go_no_settings import SCREEN_WIDTH, SCREEN_HEIGHT, PHYSICAL_WIDTH, PHYSICAL_HEIGHT, FPS, NUM_TRIALS, TASK_TIME, SPAWN_INTERVAL, LEVEL
from player import Player
import json
import datetime
import os
import random

subject = "subject_1_gamified.json"
LEVEL = 6

last_spawn_time = pygame.time.get_ticks()
spawn_count = 0  

pygame.init()
amount_of_trials = 30

# Get display info to determine the current screen height.
info = pygame.display.Info()
physical_width = PHYSICAL_WIDTH
physical_height = PHYSICAL_HEIGHT

# Set the display mode to full screen with the calculated physical dimensions.
screen = pygame.display.set_mode((physical_width, physical_height), pygame.FULLSCREEN)
clock = pygame.time.Clock()

# Create a game surface using your desired game resolution.
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

# Calculate offsets to center the game surface on the physical display.
offset_x = (physical_width - SCREEN_WIDTH) // 2
offset_y = (physical_height - SCREEN_HEIGHT) // 2

# ---------------------------------------------------------------------
# Instruction Screen Function
def show_instructions():
    # Fill the screen with a black background.
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 48)

    # Draw the welcome text at the top center.
    welcome_text = font.render("Welcome to Ninja Swipe!", True, (255, 255, 255))
    welcome_rect = welcome_text.get_rect(center=(physical_width // 2, 80))
    screen.blit(welcome_text, welcome_rect)

    # Load images.
    try:
        shuriken_img = pygame.image.load('Go-Nogo/assets/sprites/shuriken.png').convert_alpha()
        shuriken_img = pygame.transform.scale(shuriken_img, (100, 100))
    except Exception as e:
        print("Error loading shuriken image:", e)
        shuriken_img = None

    try:
        heart_img = pygame.image.load('Go-Nogo/assets/sprites/heart.png').convert_alpha()
        heart_img = pygame.transform.scale(heart_img, (100, 100))
    except Exception as e:
        print("Error loading heart image:", e)
        heart_img = None

    gap = 10

    # --- Shuriken Instruction Line ---
    # "If you see (shuriken image) Tap the Screen!"
    text1 = font.render("If you see", True, (255, 255, 255))
    text2 = font.render("Tap the Screen!", True, (255, 255, 255))
    shuriken_img_width = 100 if shuriken_img is not None else 0
    total_width = text1.get_width() + gap + shuriken_img_width + gap + text2.get_width()
    start_x = (physical_width - total_width) // 2
    y_shuriken = physical_height // 2 - 100  # Adjust vertical placement as needed

    # Blit the first part.
    text1_rect = text1.get_rect(midleft=(start_x, y_shuriken))
    screen.blit(text1, text1_rect)

    # Blit the shuriken image.
    if shuriken_img:
        image_x = text1_rect.right + gap
        image_rect = shuriken_img.get_rect(midleft=(image_x, y_shuriken))
        screen.blit(shuriken_img, image_rect)
        right_edge = image_rect.right
    else:
        right_edge = text1_rect.right

    # Blit the second part.
    text2_rect = text2.get_rect(midleft=(right_edge + gap, y_shuriken))
    screen.blit(text2, text2_rect)

    # --- Heart Instruction Line ---
    # "If you see (heart image) Don't tap the screen!"
    text3 = font.render("If you see", True, (255, 255, 255))
    text4 = font.render("Don't tap the screen!", True, (255, 255, 255))
    heart_img_width = 100 if heart_img is not None else 0
    total_width2 = text3.get_width() + gap + heart_img_width + gap + text4.get_width()
    start_x2 = (physical_width - total_width2) // 2
    y_heart = physical_height // 2 + 50  # Adjust vertical placement as needed

    text3_rect = text3.get_rect(midleft=(start_x2, y_heart))
    screen.blit(text3, text3_rect)

    if heart_img:
        image_x2 = text3_rect.right + gap
        heart_rect = heart_img.get_rect(midleft=(image_x2, y_heart))
        screen.blit(heart_img, heart_rect)
        right_edge2 = heart_rect.right
    else:
        right_edge2 = text3_rect.right

    text4_rect = text4.get_rect(midleft=(right_edge2 + gap, y_heart))
    screen.blit(text4, text4_rect)

    # --- Bottom Instruction ---
    ready_text = font.render("Are you ready? Tap the screen to continue", True, (255, 255, 255))
    ready_rect = ready_text.get_rect(center=(physical_width // 2, physical_height - 80))
    screen.blit(ready_text, ready_rect)

    pygame.display.flip()

    # Wait for the user to press a key or tap the screen.
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                waiting = False

# Show instructions before starting the main game loop.
show_instructions()
# ---------------------------------------------------------------------

correct_counter = 0
SHURIKEN_SPAWN_POINT = (SCREEN_HEIGHT//12, SCREEN_HEIGHT//2)  # Coordinates for a single spawn point

# Load and scale background
background = pygame.image.load('Go-Nogo/assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (physical_width, SCREEN_HEIGHT))

# Load and scale path while maintaining aspect ratio
path = pygame.image.load('Go-Nogo/assets/sprites/path.png').convert_alpha()
original_width, original_height = path.get_size()
TARGET_PATH_HEIGHT = SCREEN_HEIGHT  # Adjust as needed
aspect_ratio = original_width / original_height
TARGET_PATH_WIDTH = int(TARGET_PATH_HEIGHT * aspect_ratio)
path = pygame.transform.scale(path, (TARGET_PATH_WIDTH, TARGET_PATH_HEIGHT))
path_rect = path.get_rect(midleft=(SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))

prompt = ''

# Load hurt frames
hurt_frames = [
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/1.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/2.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/3.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10))
]
hurt_frame_index = 0
hurt_animation_active = False
hurt_animation_timer = 0
HURT_ANIMATION_INTERVAL = 150  # milliseconds

# Load power-up frames
powerup_frames = [
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/powerup/1.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/powerup/2.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/powerup/3.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10))
]
powerup_frame_index = 0
powerup_animation_active = False
powerup_animation_timer = 0
POWERUP_ANIMATION_INTERVAL = 150  # milliseconds

# Sprite groups
shuriken_group = pygame.sprite.Group()
player_group = pygame.sprite.GroupSingle()

# Initialize player 
player = Player(
    idle_folder='Go-Nogo/assets/sprites/player_idle', 
    bottom_left='Go-Nogo/assets/sprites/BottomLeft',
    bottom_right='Go-Nogo/assets/sprites/BottomRight',
    top_left='Go-Nogo/assets/sprites/TopLeft',
    top_right='Go-Nogo/assets/sprites/TopRight',
    left='Go-Nogo/assets/sprites/Left',
    right='Go-Nogo/assets/sprites/Right', 
    position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
)
player_group.add(player)

score = {
    "Go correct": 0,
    "Go incorrect": 0,
    "Dontgo correct": 0,
    "Dontgo incorrect": 0
}
trial_log = []

last_spawn_time = pygame.time.get_ticks()
spawn_count = 0  
HEXAGON_RADIUS = SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 10
HEXAGON_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
HEXAGON_POINTS = [
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS, HEXAGON_CENTER[1]),  # Right
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] - int(HEXAGON_RADIUS * 0.866)),  # Top-Right
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] - int(HEXAGON_RADIUS * 0.866)),  # Top-Left
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS, HEXAGON_CENTER[1]),  # Left
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] + int(HEXAGON_RADIUS * 0.866)),  # Bottom-Left
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] + int(HEXAGON_RADIUS * 0.866)),  # Bottom-Right
]

current_spawn_index = 0
feedback_score = 0
font = pygame.font.Font(None, SCREEN_HEIGHT // 20) 
feedback_message = ""  # Textual feedback message ("Great!", etc.)
feedback_timer = 0
FEEDBACK_DURATION = 800  # milliseconds

HURT_FLASH_DURATION = 300  # milliseconds
hurt_flash_active = False
hurt_flash_timer = 0

glow_active = False
glow_timer = 0
GLOW_DURATION = 300  # milliseconds

font_feedback = pygame.font.Font(None, SCREEN_HEIGHT // 20)
player_hitbox = pygame.Rect(HEXAGON_CENTER[0] - HEXAGON_RADIUS//11,
                              HEXAGON_CENTER[1] - HEXAGON_RADIUS//11, 100, 100)

last_input_time = 0
INPUT_COOLDOWN = 50  # milliseconds

# Global variable to store the outcome of the last input: "Correct" or "Incorrect"
outcome = None
input_received = False

while True:
    current_time = pygame.time.get_ticks()
    any_flying = any(shuriken.flying for shuriken in shuriken_group)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Final Score:", score)
            pygame.quit()
            sys.exit()
        # Process input only if no shuriken is flying and input hasn't been received yet.
        elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN) and not any_flying and not input_received:
            # Mark that an input was received so further ones are ignored.
            input_received = True
            
            player.slash()
            closest_shuriken = None
            min_distance = float('inf')
            for shuriken in shuriken_group:
                distance = pygame.math.Vector2(player.rect.center).distance_to(shuriken.rect.center)
                if distance < min_distance:
                    min_distance = distance
                    closest_shuriken = shuriken

            if closest_shuriken:
                destruction_time = pygame.time.get_ticks()
                if closest_shuriken.type == "Go":
                    score["Go correct"] += 1
                    closest_shuriken.slash() 
                    outcome = "Correct"
                    feedback_score += 1 
                    correct_counter += 1 
                    feedback_message = "Great!"
                    if feedback_score >= 5 and feedback_score < 10:
                        feedback_message = "Amazing!"
                    elif feedback_score >= 10:
                        feedback_message = "Unbelievable!"
                else:
                    score["Dontgo incorrect"] += 1
                    closest_shuriken.slash()  
                    hurt_animation_active = True
                    hurt_frame_index = 0
                    hurt_animation_timer = pygame.time.get_ticks()
                    outcome = "Incorrect"
                    feedback_score = feedback_score - 1 if feedback_score > 0 else 0 
                    feedback_message = "Try Again!"
                feedback_timer = pygame.time.get_ticks()
                glow_active = True
                glow_timer = pygame.time.get_ticks()
                trial_log.append({
                    "Spawn Time (s)": spawn_time / 1000,
                    "Destruction Time (s)": destruction_time / 1000,
                    "Prompt": closest_shuriken.type,
                    "Outcome": outcome,
                    'Time Spent (s)': (destruction_time - spawn_time) / 1000
                })

    # Spawn new shuriken if enough time has passed and no shurikens are on screen.
    if current_time - last_spawn_time > SPAWN_INTERVAL and len(shuriken_group) == 0:
        if spawn_count < amount_of_trials:  
            prompt = spawn_shuriken(shuriken_group, HEXAGON_POINTS, HEXAGON_CENTER, change_color=True, is_inhabitation=True)
            spawn_time = current_time
            spawn_count += 1  
            last_spawn_time = current_time
            
            if LEVEL == 7:
                SPAWN_INTERVAL = random.choice(random_times) 
            # Reset input flag for the new trial.
            input_received = False
        else:
            print("Final Score:", score)
            # Save trial log data.
            data_folder = "Go-Nogo/Validation"
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            date = datetime.datetime.now().strftime("%Y-%m-%d") 
            filename = os.path.join(data_folder, subject)
            with open(filename, "w") as json_file:
                json.dump(trial_log, json_file, indent=4)
            # Save score.
            data_folder = "Go-Nogo/Score"
            score_percent = feedback_score / NUM_TRIALS * 100
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            date = datetime.datetime.now().strftime("%Y-%m-%d") 
            filename = os.path.join(data_folder, f"{date}.json")
            with open(filename, "w") as json_file:
                json.dump(score_percent, json_file, indent=4)
            pygame.quit()
            sys.exit()

    for shuriken in shuriken_group:
        shuriken.update()
        if shuriken.rect.colliderect(player_hitbox):
            destruction_time = pygame.time.get_ticks()
            if shuriken.type == "Go":
                outcome = "Incorrect"
                score["Go incorrect"] += 1  
                feedback_score = feedback_score - 1 if feedback_score > 0 else 0 
                hurt_animation_active = True
                hurt_frame_index = 0
                hurt_animation_timer = pygame.time.get_ticks()
                feedback_message = "Try Again!"
            elif shuriken.type == "Dontgo":
                outcome = "Correct"
                correct_counter += 1
                score["Dontgo correct"] += 1
                feedback_score += 1  
                powerup_animation_active = True
                powerup_frame_index = 0
                powerup_animation_timer = pygame.time.get_ticks()
                feedback_message = "Great!"
                if feedback_score >= 5 and feedback_score < 10:
                    feedback_message = "Amazing!"
                elif feedback_score >= 10:
                    feedback_message = "Unbelievable!"
            feedback_timer = pygame.time.get_ticks()
            glow_active = True
            glow_timer = pygame.time.get_ticks()
            trial_log.append({
                "Spawn Time (s)": spawn_time / 1000,
                "Destruction Time (s)": destruction_time / 1000,
                "Prompt": shuriken.type,
                "Outcome": outcome,
                'Time Spent (s)': (destruction_time - spawn_time) / 1000
            })
            shuriken.kill()

    # --- Drawing Section ---
    # Clear the game surface with transparency.
    game_surface.fill((0, 0, 0, 0)) 
    game_surface.blit(path, path_rect)
    
    # Draw feedback (score and message).
    if current_time - feedback_timer < FEEDBACK_DURATION and outcome is not None:
        if outcome == "Correct":
            feedback_color = (0, 255, 0)
        else:
            feedback_color = (255, 0, 0)
        score_text = font_feedback.render(str(feedback_score), True, feedback_color)
        game_surface.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2,
                                       SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 6))
        message_text = font_feedback.render(feedback_message, True, feedback_color)
        game_surface.blit(message_text, (SCREEN_WIDTH // 2 - message_text.get_width() // 2,
                                         SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 8))
    
    # Draw glow effect.
    if glow_active:
        if current_time - glow_timer < GLOW_DURATION:
            GLOW_SIZE = SCREEN_HEIGHT // 4
            GLOW_RADIUS = SCREEN_HEIGHT // 14
            glow_surface = pygame.Surface((GLOW_SIZE, GLOW_SIZE), pygame.SRCALPHA)
            glow_surface.fill((0, 0, 0, 0))
            max_alpha = 255
            alpha = min(max(50 + feedback_score * 20, 50), max_alpha)
            if outcome == "Incorrect":
                glow_color = (255, 57, 20, 150)
            else:
                glow_color = (57, 255, 20, alpha)
            pygame.draw.circle(glow_surface, glow_color, (GLOW_SIZE // 2, GLOW_SIZE // 2), GLOW_RADIUS)
            glow_x = SCREEN_WIDTH // 2 - GLOW_SIZE // 2
            glow_y = SCREEN_HEIGHT // 2 - GLOW_SIZE // 2
            game_surface.blit(glow_surface, (glow_x, glow_y))
        else:
            glow_active = False

    # Update and draw sprites onto game_surface.
    player_group.update(prompt)
    shuriken_group.update()
    if not hurt_animation_active and not powerup_animation_active:
        player_group.draw(game_surface)
    shuriken_group.draw(game_surface)
    
    # Draw hurt animation if active.
    if hurt_animation_active:
        player_group.sprite.rect.y = SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 16
        if pygame.time.get_ticks() - hurt_animation_timer > HURT_ANIMATION_INTERVAL:
            hurt_frame_index += 1
            hurt_animation_timer = pygame.time.get_ticks()
            if hurt_frame_index >= len(hurt_frames):
                hurt_animation_active = False
                hurt_frame_index = 0 
        if hurt_animation_active:
            game_surface.blit(hurt_frames[hurt_frame_index], player.rect)
            
    # Draw power-up animation if active.
    if powerup_animation_active:
        if pygame.time.get_ticks() - powerup_animation_timer > POWERUP_ANIMATION_INTERVAL:
            powerup_frame_index += 1
            powerup_animation_timer = pygame.time.get_ticks()
            if powerup_frame_index >= len(powerup_frames):
                powerup_animation_active = False
                powerup_frame_index = 0 
        if powerup_animation_active:
            game_surface.blit(powerup_frames[powerup_frame_index], player.rect)
            player_group.slashing = False  

    # --- Final Blit ---
    # Draw the full-screen background.
    screen.blit(background, (0, 0))
    # Blit the centered game surface onto the screen.
    screen.blit(game_surface, (offset_x, offset_y))
    
    pygame.display.flip()

    clock.tick(FPS)
import pygame
import sys
import random
import os
import json
import datetime
from go_no_settings import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, NUM_TRIALS, LEVEL
from spawner import spawn_shuriken
from player import Player

###########################
# Tutorial Helper Functions
###########################
def show_message(screen, font, text, screen_width, screen_height, clock):
    """
    Display a multi‑line message on a black background with a "Tap to continue" prompt,
    and wait for a key or mouse tap.
    """
    screen.fill((0, 0, 0))
    lines = text.split("\n")
    line_height = font.get_height() + 10
    total_text_height = len(lines) * line_height
    y = screen_height // 2 - total_text_height // 2
    for line in lines:
        rendered_line = font.render(line, True, (255, 255, 255))
        x = screen_width // 2 - rendered_line.get_width() // 2
        screen.blit(rendered_line, (x, y))
        y += line_height
    tap_text = font.render("Tap to continue", True, (200, 200, 200))
    screen.blit(tap_text, (screen_width // 2 - tap_text.get_width() // 2, screen_height - 100))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        clock.tick(60)

def tutorial(screen, clock, font, screen_width, screen_height, idle_frames, 
             slash_frames, hurt_frames, powerup_frames, idle_frame_delay, 
             player_x, player_y, shuriken_img, heart_img, target_size):
    """
    Tutorial Phase:
      1. Welcome & introduction.
      2. Go-demo: When you see a shuriken, TAP the screen.
         Then the slash animation with "Correct!" is shown.
      3. Hurt-demo: If you don't slash, I get hurt!
         A shuriken flies toward the player, then hurt animation with "Ow!" is shown.
      4. No-go demo: When you see a heart, DO NOT tap!
         A heart flies toward the player, then power-up animation with "Awesome!" is shown.
      5. Final prompt.
    """
    # 1. Welcome
    show_message(screen, font, "Hi! Welcome to Ninja Slash!\nBefore we begin, let's do a quick test.", 
                 screen_width, screen_height, clock)
    
    # 2. Go demonstration.
    screen.fill((0, 0, 0))
    instruction = font.render("If you see a shuriken, TAP the screen!", True, (255, 255, 255))
    screen.blit(instruction, (screen_width//2 - instruction.get_width()//2, 50))
    screen.blit(idle_frames[0], (player_x, player_y))
    target_x_fixed = 50
    target_y_fixed = screen_height - 230
    screen.blit(shuriken_img, (target_x_fixed, target_y_fixed))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        clock.tick(60)
    for frame in slash_frames:
        screen.fill((0, 0, 0))
        screen.blit(frame, (player_x, player_y))
        correct_text = font.render("Correct!", True, (0, 255, 0))
        screen.blit(correct_text, (screen_width//2 - correct_text.get_width()//2,
                                   screen_height//2 - correct_text.get_height()//2))
        pygame.display.flip()
        pygame.time.delay(100)
    
    # 3. Hurt demonstration.
    show_message(screen, font, "If you don't destroy the shuriken, I get hurt!", 
                 screen_width, screen_height, clock)
    start_x = target_x_fixed
    end_x = player_x - 100
    steps = 20
    for step in range(steps):
        interp_x = start_x + (end_x - start_x) * (step / steps)
        screen.fill((0, 0, 0))
        screen.blit(shuriken_img, (interp_x, target_y_fixed))
        screen.blit(idle_frames[0], (player_x, player_y))
        pygame.display.flip()
        pygame.time.delay(50)
    for frame in hurt_frames:
        screen.fill((0, 0, 0))
        screen.blit(frame, (player_x, player_y))
        hurt_text = font.render("Ow!", True, (255, 0, 0))
        screen.blit(hurt_text, (screen_width//2 - hurt_text.get_width()//2,
                                screen_height//2 - hurt_text.get_height()//2))
        pygame.display.flip()
        pygame.time.delay(100)
    
    # 4. No-go demonstration.
    show_message(screen, font, "Now if you see a heart, DO NOT tap!\nIt is a power up!", 
                 screen_width, screen_height, clock)
    start_x = target_x_fixed
    end_x = player_x - 100
    for step in range(steps):
        interp_x = start_x + (end_x - start_x) * (step / steps)
        screen.fill((0, 0, 0))
        screen.blit(heart_img, (interp_x, target_y_fixed))
        screen.blit(idle_frames[0], (player_x, player_y))
        pygame.display.flip()
        pygame.time.delay(50)
    for frame in powerup_frames:
        screen.fill((0, 0, 0))
        screen.blit(frame, (player_x, player_y))
        powerup_text = font.render("Awesome!", True, (0, 255, 0))
        screen.blit(powerup_text, (screen_width//2 - powerup_text.get_width()//2,
                                   screen_height//2 - powerup_text.get_height()//2))
        pygame.display.flip()
        pygame.time.delay(100)
    
    # 5. Final tutorial message.
    show_message(screen, font, "Now are you ready? Here we go!", screen_width, screen_height, clock)

###########################
# Main Calibration Code
###########################
import pygame
import sys
from go_no_settings import SCREEN_WIDTH, SCREEN_HEIGHT, SHURIKEN_SPAWN_POINT, FPS, NUM_TRIALS, LEVEL
from spawner import spawn_shuriken
from player import Player
import json
import datetime
import os
import random

pygame.init()
screen_width, screen_height = 700, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Calibration Mode")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# ----- Load Assets -----
idle_frames = []
try:
    for i in range(1, 5):
        frame = pygame.image.load(f"Go-Nogo/assets/sprites/player_idle/{i}.png").convert_alpha()
        frame = pygame.transform.scale(frame, (150, 150))
        idle_frames.append(frame)
except Exception as e:
    print("Error loading idle assets:", e)
    pygame.quit(); sys.exit()

slash_frames = []
try:
    for i in range(1, 4):
        frame = pygame.image.load(f"Go-Nogo/assets/sprites/Left/{i}.png").convert_alpha()
        frame = pygame.transform.scale(frame, (150, 150))
        slash_frames.append(frame)
except Exception as e:
    print("Error loading slash assets:", e)
    pygame.quit(); sys.exit()

hurt_frames = []
try:
    for i in range(1, 4):
        frame = pygame.image.load(f"Go-Nogo/assets/sprites/hurt/{i}.png").convert_alpha()
        frame = pygame.transform.scale(frame, (150, 150))
        hurt_frames.append(frame)
except Exception as e:
    print("Error loading hurt assets:", e)
    pygame.quit(); sys.exit()

powerup_frames = []
try:
    for i in range(1, 4):
        frame = pygame.image.load(f"Go-Nogo/assets/sprites/powerup/{i}.png").convert_alpha()
        frame = pygame.transform.scale(frame, (150, 150))
        powerup_frames.append(frame)
except Exception as e:
    print("Error loading powerup assets:", e)
    pygame.quit(); sys.exit()

idle_frame_delay = 200
player_x = screen_width - 250
player_y = screen_height - 250

try:
    shuriken_img = pygame.image.load("Go-Nogo/assets/sprites/shuriken.png").convert_alpha()
    heart_img = pygame.image.load("Go-Nogo/assets/sprites/heart.png").convert_alpha()
except Exception as e:
    print("Error loading target images. Check file paths.", e)
    pygame.quit(); sys.exit()

target_size = (80, 80)
shuriken_img = pygame.transform.scale(shuriken_img, target_size)
heart_img = pygame.transform.scale(heart_img, target_size)

# ----- Tutorial Phase -----
tutorial(screen, clock, font, screen_width, screen_height, idle_frames, slash_frames, 
         hurt_frames, powerup_frames, idle_frame_delay, player_x, player_y, 
         shuriken_img, heart_img, target_size)

# Initialize calibration variables
counter = 0
LEVEL = 1

# Show initial Calibration Level message
show_message(screen, font, f"Calibration Level {LEVEL}", screen_width, screen_height, clock)

last_spawn_time = pygame.time.get_ticks()
spawn_count = 0  
amount_of_trials = NUM_TRIALS

# Load and scale background
background = pygame.image.load('Go-Nogo/assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Load and scale path while maintaining aspect ratio
path = pygame.image.load('Go-Nogo/assets/sprites/path.png').convert_alpha()
original_width, original_height = path.get_size()
TARGET_PATH_HEIGHT = SCREEN_HEIGHT
aspect_ratio = original_width / original_height
TARGET_PATH_WIDTH = int(TARGET_PATH_HEIGHT * aspect_ratio)
path = pygame.transform.scale(path, (TARGET_PATH_WIDTH, TARGET_PATH_HEIGHT))
path_rect = path.get_rect(midleft=(SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))

prompt = ''
img = pygame.image.load('Go-Nogo/assets/sprites/hurt/1.png').convert_alpha()

# Load hurt frames for in-game animation
hurt_frames = [
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/1.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/2.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('Go-Nogo/assets/sprites/hurt/3.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10))
]
hurt_frame_index = 0
hurt_animation_active = False
hurt_animation_timer = 0
HURT_ANIMATION_INTERVAL = 150  # milliseconds

# Load power-up frames for in-game animation
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
feedback_message = ""
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

# Global outcome: "Correct" or "Incorrect"
outcome = None

con = True
while con:
    current_time = pygame.time.get_ticks()
    # When 10 successful responses have been registered, upgrade level
    if counter == 10:
        LEVEL += 1
        counter = 0
        if LEVEL > 7:
            con = False  # End calibration if level > 7
        else:
            # Show new calibration level message before continuing
            show_message(screen, font, f"Calibration Level {LEVEL}", screen_width, screen_height, clock)
    # Set spawn interval based on level
    if LEVEL == 1:
        SPAWN_INTERVAL = 2500
    elif LEVEL == 2:
        SPAWN_INTERVAL = 2000
    elif LEVEL == 3:
        SPAWN_INTERVAL = 1750
    elif LEVEL == 4:
        SPAWN_INTERVAL = 1500
    elif LEVEL == 5:
        SPAWN_INTERVAL = 1250
    elif LEVEL == 6:
        SPAWN_INTERVAL = 1000
    elif LEVEL == 7:
        random_times = [300, 500, 750, 1000, 1500, 1750, 2000]
        SPAWN_INTERVAL = random.choice(random_times)
        
    any_flying = any(shuriken.flying for shuriken in shuriken_group)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Final Score:", score)
            pygame.quit()
            sys.exit()
        elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN) and not any_flying:
            if current_time - last_input_time < INPUT_COOLDOWN:
                continue
            last_input_time = current_time

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
                    counter += 1
                    feedback_message = "Great!"
                    if 5 <= feedback_score < 10:
                        feedback_message = "Amazing!"
                    elif feedback_score >= 10:
                        feedback_message = "Unbelievable!"
                else:
                    con = False  # End calibration on incorrect input
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
                    "Time Spent (s)": (destruction_time - spawn_time) / 1000
                })

    if current_time - last_spawn_time > SPAWN_INTERVAL and len(shuriken_group) == 0:
        if spawn_count < amount_of_trials:  
            prompt = spawn_shuriken(shuriken_group, HEXAGON_POINTS, HEXAGON_CENTER, change_color=True, is_inhabitation=True)
            spawn_time = current_time
            spawn_count += 1  
            last_spawn_time = current_time
            
            if LEVEL == 4:
                SPAWN_INTERVAL = random.choice(random_times) 
        else:
            print("Final Score:", score)
            data_folder = "data"
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            date = datetime.datetime.now().strftime("%Y-%m-%d") 
            filename = os.path.join(data_folder, f"go_no_go_{date}.json")
            with open(filename, "w") as json_file:
                json.dump(trial_log, json_file, indent=4)
            pygame.quit()
            sys.exit()

    for shuriken in shuriken_group:
        shuriken.update()
        if shuriken.rect.colliderect(player_hitbox):
            destruction_time = pygame.time.get_ticks()
            if shuriken.type == "Go":
                outcome = "Incorrect"
                con = False 
                score["Go incorrect"] += 1  
                feedback_score = feedback_score - 1 if feedback_score > 0 else 0 
                hurt_animation_active = True
                hurt_frame_index = 0
                hurt_animation_timer = pygame.time.get_ticks()
                feedback_message = "Try Again!"
            elif shuriken.type == "Dontgo":
                outcome = "Correct"
                counter += 1
                score["Dontgo correct"] += 1
                feedback_score += 1  
                powerup_animation_active = True
                powerup_frame_index = 0
                powerup_animation_timer = pygame.time.get_ticks()
                feedback_message = "Great!"
                if 5 <= feedback_score < 10:
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
                "Time Spent (s)": (destruction_time - spawn_time) / 1000
            })
            shuriken.kill()

    # Draw background and path
    screen.blit(background, (0, 0))
    screen.blit(path, path_rect)
    
    # Draw feedback (numeric score and message)
    if current_time - feedback_timer < FEEDBACK_DURATION and outcome is not None:
        if outcome == "Correct":
            feedback_color = (0, 255, 0)
        else:
            feedback_color = (255, 0, 0)
        score_text = font_feedback.render(str(feedback_score), True, feedback_color)
        screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 6))
        message_text = font_feedback.render(feedback_message, True, feedback_color)
        screen.blit(message_text, (SCREEN_WIDTH // 2 - message_text.get_width() // 2, SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 8))
    
    # Draw glow effect if active
    current_time = pygame.time.get_ticks()
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
            screen.blit(glow_surface, (glow_x, glow_y))
        else:
            glow_active = False

    # Update and draw sprites
    player_group.update(prompt)                    
    shuriken_group.update()
    if not hurt_animation_active and not powerup_animation_active:
        player_group.draw(screen)
    shuriken_group.draw(screen)

    # Hurt animation display
    if hurt_animation_active:
        player_group.sprite.rect.y = SCREEN_HEIGHT // 2 - SCREEN_HEIGHT // 16
        if pygame.time.get_ticks() - hurt_animation_timer > HURT_ANIMATION_INTERVAL:
            hurt_frame_index += 1
            hurt_animation_timer = pygame.time.get_ticks()
            if hurt_frame_index >= len(hurt_frames):
                hurt_animation_active = False
                hurt_frame_index = 0 
        if hurt_animation_active:
            screen.blit(hurt_frames[hurt_frame_index], player.rect)
            
    # Power-up animation display
    if powerup_animation_active:
        if pygame.time.get_ticks() - powerup_animation_timer > POWERUP_ANIMATION_INTERVAL:
            powerup_frame_index += 1
            powerup_animation_timer = pygame.time.get_ticks()
            if powerup_frame_index >= len(powerup_frames):
                powerup_animation_active = False
                powerup_frame_index = 0 
        if powerup_animation_active:
            screen.blit(powerup_frames[powerup_frame_index], player.rect)
            player_group.slashing = False  

    pygame.display.flip()
    clock.tick(FPS)

# Calibration is complete; save final level to a file
with open("Go-Nogo/calibration.txt", "w") as f:
    f.write(str(LEVEL))

# Display the final message: Calibration Complete! and the final level
screen.fill((0, 0, 0))
complete_text = font.render("Calibration Complete!", True, (255, 255, 255))
level_text = font.render("Your Level: " + str(LEVEL), True, (255, 255, 255))
screen.blit(complete_text, (screen_width//2 - complete_text.get_width()//2,
                            screen_height//2 - complete_text.get_height()//2 - 30))
screen.blit(level_text, (screen_width//2 - level_text.get_width()//2,
                            screen_height//2 - level_text.get_height()//2 + 30))
pygame.display.flip()
pygame.time.delay(3000)
pygame.quit()
sys.exit()

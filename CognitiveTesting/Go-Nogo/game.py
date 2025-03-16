import pygame
import sys
from spawner import spawn_shuriken
from go_no_settings import SCREEN_WIDTH, SCREEN_HEIGHT, PHYSICAL_WIDTH, PHYSICAL_HEIGHT, FPS, NUM_TRIALS, TASK_TIME, SPAWN_INTERVAL, LEVEL
from player import Player
import json
import datetime
import os
import random
import requests  # For HTTP requests
import time      # Standard time module for timestamps
import yaml      # For reading config.yaml

def show_instructions():
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 48)
    welcome_text = font.render("Welcome to Ninja Swipe!", True, (255, 255, 255))
    welcome_rect = welcome_text.get_rect(center=(physical_width // 2, 80))
    screen.blit(welcome_text, welcome_rect)

    try:
        shuriken_img = pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/shuriken.png').convert_alpha()
        shuriken_img = pygame.transform.scale(shuriken_img, (100, 100))
    except Exception as e:
        print("Error loading shuriken image:", e)
        shuriken_img = None

    try:
        heart_img = pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/heart.png').convert_alpha()
        heart_img = pygame.transform.scale(heart_img, (100, 100))
    except Exception as e:
        print("Error loading heart image:", e)
        heart_img = None

    gap = 10

    text1 = font.render("If you see", True, (255, 255, 255))
    text2 = font.render("Tap the Screen!", True, (255, 255, 255))
    shuriken_img_width = 100 if shuriken_img is not None else 0
    total_width = text1.get_width() + gap + shuriken_img_width + gap + text2.get_width()
    start_x = (physical_width - total_width) // 2
    y_shuriken = physical_height // 2 - 100

    text1_rect = text1.get_rect(midleft=(start_x, y_shuriken))
    screen.blit(text1, text1_rect)

    if shuriken_img:
        image_x = text1_rect.right + gap
        image_rect = shuriken_img.get_rect(midleft=(image_x, y_shuriken))
        screen.blit(shuriken_img, image_rect)
        right_edge = image_rect.right
    else:
        right_edge = text1_rect.right

    text2_rect = text2.get_rect(midleft=(right_edge + gap, y_shuriken))
    screen.blit(text2, text2_rect)

    text3 = font.render("If you see", True, (255, 255, 255))
    text4 = font.render("Don't tap the screen!", True, (255, 255, 255))
    heart_img_width = 100 if heart_img is not None else 0
    total_width2 = text3.get_width() + gap + heart_img_width + gap + text4.get_width()
    start_x2 = (physical_width - total_width2) // 2
    y_heart = physical_height // 2 + 50

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

    ready_text = font.render("Are you ready? Tap the screen to continue", True, (255, 255, 255))
    ready_rect = ready_text.get_rect(center=(physical_width // 2, physical_height - 80))
    screen.blit(ready_text, ready_rect)

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
                waiting = False

class SlashSprite(pygame.sprite.Sprite):
    def __init__(self, start_pos, target_sprite, direction, speed=15):
        super().__init__()
        self.image = pygame.image.load(f'CognitiveTesting/Go-Nogo/assets/sprites/slash/{direction}.png').convert_alpha()
        target_size = target_sprite.image.get_size()
        self.image = pygame.transform.scale(self.image, target_size)
        self.rect = self.image.get_rect(center=start_pos)
        self.target_sprite = target_sprite
        self.target_pos = pygame.math.Vector2(target_sprite.rect.center)
        self.position = pygame.math.Vector2(start_pos)
        direction_vector = self.target_pos - self.position
        if direction_vector.length() != 0:
            self.velocity = direction_vector.normalize() * speed
        else:
            self.velocity = pygame.math.Vector2(0, 0)

    def update(self):
        global score, outcome, feedback_score, correct_counter, feedback_message, feedback_timer, glow_active, glow_timer, trial_log, spawn_time, hurt_animation_active, hurt_frame_index, hurt_animation_timer, input_press_time, game_client
        self.position += self.velocity
        self.rect.center = self.position
        if self.position.distance_to(self.target_pos) < self.velocity.length():
            destruction_time = input_press_time
            if self.target_sprite.type == "Go":
                score["Go correct"] += 1
                self.target_sprite.slash() 
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
                self.target_sprite.slash()
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
                "Prompt": self.target_sprite.type,
                "Outcome": outcome,
                'Time Spent (s)': (destruction_time - spawn_time) / 1000
            })
            self.kill()

last_spawn_time = pygame.time.get_ticks()
spawn_count = 0  

pygame.init()
amount_of_trials = NUM_TRIALS
info = pygame.display.Info()
physical_width = PHYSICAL_WIDTH
physical_height = PHYSICAL_HEIGHT

screen = pygame.display.set_mode((physical_width, physical_height), pygame.FULLSCREEN)
clock = pygame.time.Clock()
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

offset_x = (physical_width - SCREEN_WIDTH) // 2
offset_y = (physical_height - SCREEN_HEIGHT) // 2

show_instructions()

correct_counter = 0
random_times = [300, 500, 750, 1000, 1500, 1750, 2000]
SHURIKEN_SPAWN_POINT = (SCREEN_HEIGHT//12, SCREEN_HEIGHT//2) 
background = pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (physical_width, SCREEN_HEIGHT))
path = pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/path.png').convert_alpha()
original_width, original_height = path.get_size()
TARGET_PATH_HEIGHT = SCREEN_HEIGHT
aspect_ratio = original_width / original_height
TARGET_PATH_WIDTH = int(TARGET_PATH_HEIGHT * aspect_ratio)
path = pygame.transform.scale(path, (TARGET_PATH_WIDTH, TARGET_PATH_HEIGHT))
path_rect = path.get_rect(midleft=(SCREEN_WIDTH // 15, SCREEN_HEIGHT // 2))
prompt = ''

hurt_frames = [
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/hurt/1.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/hurt/2.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/hurt/3.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10))
]
hurt_frame_index = 0
hurt_animation_active = False
hurt_animation_timer = 0
HURT_ANIMATION_INTERVAL = 150 

# Load power-up frames
powerup_frames = [
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/powerup/1.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/powerup/2.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10)),
    pygame.transform.scale(pygame.image.load('CognitiveTesting/Go-Nogo/assets/sprites/powerup/3.png').convert_alpha(), (SCREEN_HEIGHT//9, SCREEN_HEIGHT//10))
]
powerup_frame_index = 0
powerup_animation_active = False
powerup_animation_timer = 0
POWERUP_ANIMATION_INTERVAL = 150  # milliseconds

# Sprite groups
shuriken_group = pygame.sprite.Group()
player_group = pygame.sprite.GroupSingle()
slash_group = pygame.sprite.Group()  # New group for flying slash sprites

# Initialize player 
player = Player(
    idle_folder='CognitiveTesting/Go-Nogo/assets/sprites/player_idle', 
    bottom_left='CognitiveTesting/Go-Nogo/assets/sprites/BottomLeft',
    bottom_right='CognitiveTesting/Go-Nogo/assets/sprites/BottomRight',
    top_left='CognitiveTesting/Go-Nogo/assets/sprites/TopLeft',
    top_right='CognitiveTesting/Go-Nogo/assets/sprites/TopRight',
    left='CognitiveTesting/Go-Nogo/assets/sprites/Left',
    right='CognitiveTesting/Go-Nogo/assets/sprites/Right', 
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
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS, HEXAGON_CENTER[1]),
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] - int(HEXAGON_RADIUS * 0.866)),
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] - int(HEXAGON_RADIUS * 0.866)),
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS, HEXAGON_CENTER[1]),
    (HEXAGON_CENTER[0] - HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] + int(HEXAGON_RADIUS * 0.866)),
    (HEXAGON_CENTER[0] + HEXAGON_RADIUS // 2, HEXAGON_CENTER[1] + int(HEXAGON_RADIUS * 0.866)),
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

# Global variable to store the outcome of the last input.
outcome = None
input_received = False

while True:
    current_time = pygame.time.get_ticks()
    any_flying = any(shuriken.flying for shuriken in shuriken_group)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("Final Score:", score)
            # Calculate a “score string” or however else you want to measure performance.
            final_correct = score["Go correct"] + score["Dontgo correct"]
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            score_str = f"{final_correct}/{NUM_TRIALS}"   # e.g. "7/10"


            pygame.quit()
            sys.exit()
        elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
            if not any_flying and not input_received:
                input_received = True
                global input_press_time
                input_press_time = pygame.time.get_ticks()
                player.slash()
                closest_shuriken = None
                min_distance = float('inf')
                for shuriken in shuriken_group:
                    distance = pygame.math.Vector2(player.rect.center).distance_to(shuriken.rect.center)
                    if distance < min_distance:
                        min_distance = distance
                        closest_shuriken = shuriken
                if closest_shuriken:
                    slash_sprite = SlashSprite(player.rect.center, closest_shuriken, prompt)
                    slash_group.add(slash_sprite)
    
    if current_time - last_spawn_time > SPAWN_INTERVAL and len(shuriken_group) == 0:
        if spawn_count < amount_of_trials:  
            prompt = spawn_shuriken(shuriken_group, HEXAGON_POINTS, HEXAGON_CENTER, change_color=True, is_inhabitation=True)
            spawn_time = current_time
            spawn_count += 1  
            last_spawn_time = current_time
            
            if LEVEL == 7:
                SPAWN_INTERVAL = random.choice(random_times)
            input_received = False
        else:
            print("Final Score:", score)
            data_folder = "CognitiveTesting/Go-Nogo/data"
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            date_str = datetime.datetime.now().strftime("%Y-%m-%d") 
            filename = os.path.join(data_folder, f"go_no_go_{date_str}.json")
            with open(filename, "w") as json_file:
                json.dump(trial_log, json_file, indent=4)
            data_folder = "CognitiveTesting/Go-Nogo/Score"
            score_percent = feedback_score / NUM_TRIALS * 100
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)
            filename = os.path.join(data_folder, f"{date_str}.json")
            with open(filename, "w") as json_file:
                json.dump(score_percent, json_file, indent=4)

            pygame.quit()
            sys.exit()

    for shuriken in shuriken_group:
        shuriken.update()
        if shuriken.rect.colliderect(player_hitbox):
            destruction_time = input_press_time
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
    game_surface.fill((0, 0, 0, 0))
    game_surface.blit(path, path_rect)
    
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

    player_group.update(prompt)
    shuriken_group.update()
    slash_group.update()
    if not hurt_animation_active and not powerup_animation_active:
        player_group.draw(game_surface)
    shuriken_group.draw(game_surface)
    slash_group.draw(game_surface)
    
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
            
    if powerup_animation_active:
        if pygame.time.get_ticks() - powerup_animation_timer > POWERUP_ANIMATION_INTERVAL:
            powerup_frame_index += 1
            powerup_animation_timer = pygame.time.get_ticks()
            if powerup_frame_index >= len(powerup_frames):
                powerup_animation_active = False
                powerup_frame_index = 0 
        if powerup_animation_active:
            game_surface.blit(powerup_frames[powerup_frame_index], player.rect)
            player_group.sprite.slashing = False  

    screen.blit(background, (0, 0))
    screen.blit(game_surface, (offset_x, offset_y))
    
    pygame.display.flip()

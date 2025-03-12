import pygame
import sys
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, SHURIKEN_SPAWN_POINT, FPS
from spawner import spawn_shuriken, Shuriken
from player import Player
import json

pygame.init()
amount_of_trials = 20
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
background = pygame.image.load('assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
path = pygame.image.load('assets/sprites/path.png').convert_alpha()
path_rect = path.get_rect(midleft=(50, SCREEN_HEIGHT // 2))
prompt = ''
hurt_frames = [
    pygame.image.load('assets/sprites/hurt/1.png').convert_alpha(),
    pygame.image.load('assets/sprites/hurt/2.png').convert_alpha(),
    pygame.image.load('assets/sprites/hurt/3.png').convert_alpha()
]
hurt_frame_index = 0
hurt_animation_active = False
hurt_animation_timer = 0
#Time Interval
HURT_ANIMATION_INTERVAL = 150  
powerup_frames = [
    pygame.image.load('assets/sprites/powerup/1.png').convert_alpha(),
    pygame.image.load('assets/sprites/powerup/2.png').convert_alpha(),
    pygame.image.load('assets/sprites/powerup/3.png').convert_alpha(),
]
powerup_frame_index = 0
powerup_animation_active = False
powerup_animation_timer = 0
POWERUP_ANIMATION_INTERVAL = 150  
shuriken_group = pygame.sprite.Group()
player_group = pygame.sprite.GroupSingle()

# Initialize player 
player = Player(idle_folder='assets/sprites/player_idle', 
                bottom_left='assets/sprites/BottomLeft',
                bottom_right='assets/sprites/BottomRight',
                top_left='assets/sprites/TopLeft',
                top_right='assets/sprites/TopRight',
                left='assets/sprites/Left',
                right='assets/sprites/Right', 
                position=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
player_group.add(player)
score = {
    "Go correct": 0,
    "Go incorrect": 0,
    "Dontgo correct": 0,
    "Dontgo incorrect": 0
}
trial_log = []

SPAWN_INTERVAL = 1000  
last_spawn_time = pygame.time.get_ticks()
spawn_count = 0  
HEXAGON_RADIUS = SCREEN_HEIGHT // 2 
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
FEEDBACK_DURATION = 800 

HURT_FLASH_DURATION = 300  # Time (in ms) for red flash
hurt_flash_active = False  # Whether red glow is active
hurt_flash_timer = 0  # Timer for tracking duration

glow_active = False  # Whether the glow effect should be drawn
glow_timer = 0       # The time (in ms) when the glow was activated
GLOW_DURATION = 300  # Duration in milliseconds for which the glow should display

font_feedback = pygame.font.Font(None, SCREEN_HEIGHT // 20)
player_hitbox = pygame.Rect(HEXAGON_CENTER[0] - HEXAGON_RADIUS//11, HEXAGON_CENTER[1] - HEXAGON_RADIUS//11, 100, 100)

while True:
    
    # Draw the background and path
    screen.blit(background, (0, 0))
    screen.blit(path, path_rect)

    # Update the display
    pygame.display.flip()
    clock.tick(FPS)
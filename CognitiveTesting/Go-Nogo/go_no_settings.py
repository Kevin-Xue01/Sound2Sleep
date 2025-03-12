# settings.py
import pygame
import os
import random

FPS = 60
NUM_TRIALS = 5
TASK_TIME = 1500  # in milliseconds

# Initialize pygame (or at least its display module)
pygame.init()

# Get physical display dimensions
info = pygame.display.Info()

PHYSICAL_WIDTH = info.current_w
PHYSICAL_HEIGHT = info.current_h
SCREEN_HEIGHT = PHYSICAL_HEIGHT
SCREEN_WIDTH = int(SCREEN_HEIGHT * (7 / 6))

# Attempt to read the calibration level from "calibration_level.txt"
calibration_file = "calibration_level.txt"
if os.path.exists(calibration_file):
    try:
        with open(calibration_file, "r") as f:
            LEVEL = int(f.read().strip())
    except Exception as e:
        print("Error reading calibration level:", e)
        LEVEL = 4  # default level on error
else:
    LEVEL = 4  # default level if file doesn't exist

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
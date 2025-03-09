# settings.py
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 600
SHURIKEN_SPAWN_POINT = (SCREEN_HEIGHT//12, SCREEN_HEIGHT//2)  # Coordinates for a single spawn point
FPS = 60
NUM_TRIALS = 30
TASK_TIME = 1500  # Time in milliseconds for each trial

import os
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
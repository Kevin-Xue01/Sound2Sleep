# settings.py
import pygame
import os
import random

# Initialize pygame (or at least its display module)
pygame.init()

# Get physical display dimensions
info = pygame.display.Info()

PHYSICAL_WIDTH = info.current_w
PHYSICAL_HEIGHT = info.current_h
SCREEN_HEIGHT = PHYSICAL_HEIGHT
SCREEN_WIDTH = int(SCREEN_HEIGHT * (12 / 9))

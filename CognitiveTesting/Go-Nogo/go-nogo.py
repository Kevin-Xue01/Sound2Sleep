import pygame
import random
import os
import datetime
import json

# Subject
subject = "subject_1_standard.json"

# Create the data directory if it doesn't exist
data_dir = os.path.join("Go-Nogo", "Validation")
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, subject)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Go/No-Go Touchscreen Task")
clock = pygame.time.Clock()

# Define colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)

# Function to show instructions before starting the task
def show_instructions():
    screen.fill(WHITE)
    instructions = [
        "Welcome to the Go/No-Go Task.",
        "Instructions:",
        "1. When you see a GREEN CIRCLE, TAP the screen.",
        "2. When you see a RED SQUARE, do NOT tap.",
        "Try to respond quickly and accurately.",
        "Tap anywhere to start."
    ]
    font = pygame.font.SysFont(None, 36)
    y_offset = 150
    for line in instructions:
        text = font.render(line, True, BLACK)
        rect = text.get_rect(center=(400, y_offset))
        screen.blit(text, rect)
        y_offset += 40
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False

# Function to draw the stimulus
def draw_stimulus(stimulus_type):
    screen.fill(WHITE)
    if stimulus_type == "go":
        # Draw a green circle for "go"
        pygame.draw.circle(screen, GREEN, (400, 300), 50)
    elif stimulus_type == "no-go":
        # Draw a red square for "no-go"
        pygame.draw.rect(screen, RED, (350, 250, 100, 100))
    pygame.display.flip()

# Function to display feedback for 2 seconds on incorrect trials
def show_feedback(feedback_text):
    screen.fill(WHITE)
    font = pygame.font.SysFont(None, 48)
    text = font.render(feedback_text, True, BLACK)
    rect = text.get_rect(center=(400, 300))
    screen.blit(text, rect)
    pygame.display.flip()
    pygame.time.delay(2000)  # 2 second feedback

# Show instructions before starting
show_instructions()

trial_num = 0
trials_data = []  # List to hold trial dictionaries
total_trials = 30  # Adjust number of trials as needed
running = True

while running and trial_num < total_trials:
    trial_num += 1
    # Randomly choose stimulus type with 80% "go" and 20% "no-go"
    stimulus_type = random.choices(["go", "no-go"], weights=[80, 20])[0]
    draw_stimulus(stimulus_type)
    trial_start = pygame.time.get_ticks()
    
    response = None
    response_time = None
    responded = False
    # Maximum wait time per trial (2000 ms)
    while pygame.time.get_ticks() - trial_start < 2000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.MOUSEBUTTONDOWN:
                response = "touched"
                response_time = pygame.time.get_ticks() - trial_start
                responded = True
                break
        if responded:
            break

    # Determine if the response was correct:
    # For "go" trials, a touch is correct; for "no-go" trials, no touch is correct.
    if stimulus_type == "go":
        correct = response is not None
    else:
        correct = response is None

    # Only prepare a feedback message for incorrect trials
    if not correct:
        if stimulus_type == "go":
            feedback = "Press on the green circle!"
        else:
            feedback = "Do not press on the red square!"
    else:
        feedback = None

    # Log trial data as a dictionary
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trial_data = {
        "trial": trial_num,
        "stimulus": stimulus_type,
        "response": response/100,
        "response_time": response_time,
        "correct": correct,
        "timestamp": timestamp
    }
    trials_data.append(trial_data)
    print(trial_data)

    # If the trial was incorrect, display feedback for 2 seconds.
    # Otherwise, show a blank screen during the inter-trial interval.
    if feedback:
        show_feedback(feedback)
    else:
        screen.fill(WHITE)
        pygame.display.flip()
        pygame.time.delay(500)  # 500 ms inter-trial interval

pygame.quit()

# Save the data to a JSON file
with open(data_file, "w") as f:
    json.dump(trials_data, f, indent=4)

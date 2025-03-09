import pygame
import sys
import os
import random
import json

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Load assets
background = pygame.image.load('assets/vpat/background.jpg').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
icon_path = 'assets/vpat/icons'
icons = [pygame.image.load(os.path.join(icon_path, file)).convert_alpha() for file in os.listdir(icon_path)]
icons = [pygame.transform.scale(icon, (70, 70)) for icon in icons]

# Generate pairs
random.shuffle(icons)
pairs = [
    ({"surface": icons[i], "path": os.path.join(icon_path, os.listdir(icon_path)[i])},
     {"surface": icons[i + 1], "path": os.path.join(icon_path, os.listdir(icon_path)[i + 1])})
    for i in range(0, 20, 2)
]

# Game variables
pair_index = 0
matching_game_active = False
random_icons = []
target_icon = None
incorrect_choices = []
matched_pairs = []
attempts = 0  # Track the number of attempts for the current pair

# Feedback settings
feedback_message = ""
feedback_start_time = None
FEEDBACK_DURATION = 1000  # 1 second for feedback display

# Display functions
def display_pair(pair):
    """Display a pair of icons on the screen."""
    left_icon = pair[0]["surface"]
    right_icon = pair[1]["surface"]
    left_rect = left_icon.get_rect(center=(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 250))
    right_rect = right_icon.get_rect(center=(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 + 250))
    screen.blit(left_icon, left_rect)
    screen.blit(right_icon, right_rect)

def display_feedback():
    """Display feedback text on the screen."""
    if feedback_message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(feedback_message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

def display_matching_game(target_icon, random_icons, selected_index, incorrect_choices):
    """Display the target icon and 10 random icons in a 5x2 grid, marking incorrect choices with a red X."""
    # Display target icon and hovered icon side by side
    box_width, box_height = 100, 100
    spacing = 20
    target_box_rect = pygame.Rect(SCREEN_WIDTH // 2 - box_width - spacing, SCREEN_HEIGHT // 2 + 50, box_width, box_height)
    hovered_box_rect = pygame.Rect(SCREEN_WIDTH // 2 + spacing, SCREEN_HEIGHT // 2 + 50, box_width, box_height)

    # Draw boxes
    pygame.draw.rect(screen, (255, 255, 255), target_box_rect, 3)
    pygame.draw.rect(screen, (255, 255, 255), hovered_box_rect, 3)

    # Blit target icon
    target_icon_rect = target_icon["surface"].get_rect(center=target_box_rect.center)
    screen.blit(target_icon["surface"], target_icon_rect)

    # Blit hovered icon
    hovered_icon = random_icons[selected_index]["surface"]
    hovered_icon_rect = hovered_icon.get_rect(center=hovered_box_rect.center)
    screen.blit(hovered_icon, hovered_icon_rect)

    # Grid settings
    grid_cols, grid_rows = 5, 2
    box_width, box_height = 80, 80
    box_spacing = 10
    grid_start_x = (SCREEN_WIDTH - (box_width + box_spacing) * grid_cols) // 2
    grid_start_y = SCREEN_HEIGHT - 200

    for i, icon in enumerate(random_icons):
        row = i // grid_cols
        col = i % grid_cols
        x_pos = grid_start_x + col * (box_width + box_spacing)
        y_pos = grid_start_y + row * (box_height + box_spacing)
        box_rect = pygame.Rect(x_pos, y_pos, box_width, box_height)

        # Highlight selected box
        if i == selected_index:
            pygame.draw.rect(screen, (0, 255, 0), box_rect, 3)
        else:
            pygame.draw.rect(screen, (255, 255, 255), box_rect, 3)

        # Blit the icon inside the box
        icon_rect = icon["surface"].get_rect(center=box_rect.center)
        screen.blit(icon["surface"], icon_rect)

        # Draw a red "X" on boxes that were chosen incorrectly
        if i in incorrect_choices:
            font = pygame.font.SysFont(None, 70)
            text = font.render("X", True, (255, 0, 0))
            screen.blit(text, text.get_rect(center=box_rect.center))

def display_matched_pairs():
    """Display matched pairs dynamically relative to the screen size."""
    screen_width, screen_height = screen.get_size()
    pair_width = screen_width // 6  # Divide screen width into 6 sections for spacing
    pair_height = screen_height // 10  # Adjust row spacing dynamically

    for i, (pair, was_correct) in enumerate(matched_pairs):
        # Determine row and column
        row = i // 5  # 5 pairs per row
        col = i % 5   # Column index (0 to 4)

        # Calculate positions based on row and column
        x_offset = (col + 1) * pair_width - 40  # Adjust horizontal placement
        y_offset = (row + 1) * pair_height  # Adjust vertical placement

        left_icon = pygame.transform.scale(pair[0]["surface"], (40, 40))
        right_icon = pygame.transform.scale(pair[1]["surface"], (40, 40))
        left_rect = left_icon.get_rect(topleft=(x_offset, y_offset))
        right_rect = right_icon.get_rect(topleft=(x_offset + 40, y_offset))

        # Draw a box around the pair
        box_color = (0, 0, 255) if was_correct else (255, 0, 0)
        pygame.draw.rect(screen, box_color, left_rect.union(right_rect).inflate(10, 10), 3)

        # Draw the icons
        screen.blit(left_icon, left_rect)
        screen.blit(right_icon, right_rect)

# Game loop
running = True
random_icons = []
incorrect_choices = []
selected_box_index = 0

# Store game results
game_results = []

# Initialize game state
game_state = "showing_pairs"  # Possible states: 'showing_pairs', 'matching_game'
message = "Remember these pairs!"  # Initial message to display
message_start_time = pygame.time.get_ticks()  # Record when the message starts
MESSAGE_DURATION = 1200  # Duration in milliseconds to display the message

while running:
    screen.blit(background, (0, 0))
    display_matched_pairs()

    if game_state == "showing_pairs":
        if pair_index < len(pairs):
            display_pair(pairs[pair_index])
        else:
            game_state = "matching_game"
            message = "Now Match the Pairs!"
            message_start_time = pygame.time.get_ticks()  # Reset the message timer
            target_icon = pairs[0][0]  # Use the first icon of the pair as the target
            correct_icon = pairs[0][1]  # Use the second icon as the correct answer
            available_icons = [
                {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                for icon in icons
                if icon != correct_icon["surface"]
            ]
            random_icons = random.sample(available_icons, 9) + [correct_icon]
            random.shuffle(random_icons)
            incorrect_choices = []
            attempts = 0
    elif game_state == "matching_game":
        display_matching_game(target_icon, random_icons, selected_box_index, incorrect_choices)

    # Display the current message if within the duration
    if pygame.time.get_ticks() - message_start_time < MESSAGE_DURATION and message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT//2)))

    display_feedback()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if game_state == "showing_pairs" and event.key == pygame.K_SPACE:
                pair_index += 1
            elif game_state == "matching_game":
                if event.key == pygame.K_RIGHT:
                    selected_box_index = (selected_box_index + 1) % len(random_icons)
                elif event.key == pygame.K_LEFT:
                    selected_box_index = (selected_box_index - 1) % len(random_icons)
                elif event.key == pygame.K_DOWN:
                    selected_box_index = (selected_box_index + 5) % len(random_icons)
                elif event.key == pygame.K_UP:
                    selected_box_index = (selected_box_index - 5) % len(random_icons)
                elif event.key == pygame.K_RETURN:
                    attempts += 1
                    if random_icons[selected_box_index]["path"] == correct_icon["path"]:
                        feedback_message = "Correct!"
                        message = "Correct!"
                        message_start_time = pygame.time.get_ticks()  # Reset the message timer
                        matched_pairs.append(((target_icon, correct_icon), True))
                        game_results.append({
                            "prompt": [target_icon["path"], correct_icon["path"]],
                            "number_of_trials": attempts,
                            "correct_or_incorrect": "Correct"
                        })
                        if len(matched_pairs) < len(pairs):
                            target_icon = pairs[len(matched_pairs)][0]
                            correct_icon = pairs[len(matched_pairs)][1]
                            available_icons = [
                                {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                                for icon in icons
                                if icon != correct_icon["surface"]
                            ]
                            random_icons = random.sample(available_icons, 9) + [correct_icon]
                            random.shuffle(random_icons)
                            incorrect_choices = []
                            attempts = 0
                        else:
                            feedback_message = "All pairs matched!"
                            running = False
                    else:
                        feedback_message = "Try Again!"
                        message = "Try Again!"
                        message_start_time = pygame.time.get_ticks()  # Reset the message timer
                        incorrect_choices.append(selected_box_index)
                        if attempts >= 5:
                            matched_pairs.append(((target_icon, correct_icon), False))
                            game_results.append({
                                "prompt": [target_icon["path"], correct_icon["path"]],
                                "number_of_trials": attempts,
                                "correct_or_incorrect": "Incorrect"
                            })
                            if len(matched_pairs) < len(pairs):
                                target_icon = pairs[len(matched_pairs)][0]
                                correct_icon = pairs[len(matched_pairs)][1]
                                available_icons = [
                                    {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                                    for icon in icons
                                    if icon != correct_icon["surface"]
                                ]
                                random_icons = random.sample(available_icons, 9) + [correct_icon]
                                random.shuffle(random_icons)
                                incorrect_choices = []
                                attempts = 0
                            else:
                                feedback_message = "All pairs processed!"
                                running = False

    if feedback_message and pygame.time.get_ticks() - (feedback_start_time or 0) > FEEDBACK_DURATION:
        feedback_message = ""

    pygame.display.flip()
    clock.tick(FPS)

# Save results to JSON
with open("game_results.json", "w") as f:
    json.dump(game_results, f, indent=4)

pygame.quit()
sys.exit()

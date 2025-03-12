import pygame
import sys
import os
import random

counter = 0
# -------------------------------
# Helper Function: Show Message
# -------------------------------
def show_message(screen, font, text, screen_width, screen_height, clock):
    """
    Display a multi‑line message on a black background with a "Tap to continue" prompt.
    Waits for a key or mouse tap before returning.
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
    prompt = font.render("Tap to continue", True, (200, 200, 200))
    screen.blit(prompt, (screen_width // 2 - prompt.get_width() // 2, screen_height - 100))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        clock.tick(60)

# -------------------------------
# Function: Display a Pair and Wait
# -------------------------------
def display_pair(screen, font, pair, screen_width, screen_height, clock):
    """
    Display a pair of icons side by side and wait for a tap.
    """
    screen.fill((0, 0, 0))
    left_icon = pair[0]["surface"]
    right_icon = pair[1]["surface"]
    left_rect = left_icon.get_rect(center=(screen_width // 2 - 100, screen_height // 2))
    right_rect = right_icon.get_rect(center=(screen_width // 2 + 100, screen_height // 2))
    screen.blit(left_icon, left_rect)
    screen.blit(right_icon, right_rect)
    label = font.render("Memorize this pair", True, (255, 255, 255))
    screen.blit(label, (screen_width // 2 - label.get_width() // 2, screen_height // 2 - 150))
    pygame.display.flip()
    
    # Wait for a key or mouse tap so you have time to view the pair.
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                waiting = False
        clock.tick(60)

# -------------------------------
# Function: Matching Tutorial
# -------------------------------
def matching_tutorial(screen, clock, font, screen_width, screen_height, icons, icon_path, pairs):
    """
    Tutorial Phase:
      1. Show starting message & instructions.
      2. Display one pair for memorization (waits for tap).
      3. Display matching challenge: the player chooses which option completes the pair.
         Displays "Correct!" or "Try Again!" based on your selection.
    """
    # Step 1: Welcome and instructions
    welcome_text = ("Welcome to Matching Pairs!\n"
                    "In this tutorial, \nyou will first memorize one pair.")
    show_message(screen, font, welcome_text, screen_width, screen_height, clock)
    
    show_message(screen, font, "Tap to continue once \nyou've memorized the pair.", screen_width, screen_height, clock)
    
    # Step 2: Display the pair and wait until you tap
    display_pair(screen, font, pairs[0], screen_width, screen_height, clock)
    
    # Step 3: Matching challenge instructions
    instructions = ("Now, try to identify which of the options\n"
                    "matches the pair for this image.\n"
                    "Use the LEFT/RIGHT arrow keys to move\n"
                    "and press ENTER to choose.")
    show_message(screen, font, instructions, screen_width, screen_height, clock)
    
    # Set up matching trial using the first pair
    target_icon = pairs[0][0]
    correct_icon = pairs[0][1]
    
    # Build grid of options (level 0: total 3 options: one correct and two distractors)
    available_icons = []
    for file in os.listdir(icon_path):
        icon_img = pygame.image.load(os.path.join(icon_path, file)).convert_alpha()
        icon_img = pygame.transform.scale(icon_img, (70, 70))
        available_icons.append({"surface": icon_img, "path": os.path.join(icon_path, file)})
    # Remove the correct one from distractors
    available_icons = [icon for icon in available_icons if icon["path"] != correct_icon["path"]]
    distractors = random.sample(available_icons, 2)
    options = distractors + [correct_icon]
    random.shuffle(options)
    
    selected_index = 0
    incorrect_choices = []
    decision_message = ""
    selecting = True

    while selecting:
        screen.fill((0, 0, 0))
        # Display the target icon at the top with label "Target"
        target_rect = pygame.Rect(screen_width//2 - 150, screen_height//2 - 250, 100, 100)
        pygame.draw.rect(screen, (255, 255, 255), target_rect, 3)
        screen.blit(target_icon["surface"], (target_rect.x + 15, target_rect.y + 15))
        target_label = font.render("Target", True, (0, 255, 0))
        screen.blit(target_label, (target_rect.x, target_rect.y - 40))
        
        # Display grid of options in one row
        grid_cols = len(options)
        box_width, box_height = 100, 100
        box_spacing = 20
        grid_start_x = screen_width // 2 - ((box_width + box_spacing) * grid_cols) // 2
        grid_start_y = screen_height // 2 + 100
        for i, option in enumerate(options):
            x_pos = grid_start_x + i * (box_width + box_spacing)
            box_rect = pygame.Rect(x_pos, grid_start_y, box_width, box_height)
            if i == selected_index:
                pygame.draw.rect(screen, (0, 255, 0), box_rect, 3)
            else:
                pygame.draw.rect(screen, (255, 255, 255), box_rect, 3)
            icon_rect = option["surface"].get_rect(center=box_rect.center)
            screen.blit(option["surface"], icon_rect)
            if i in incorrect_choices:
                x_mark = font.render("X", True, (255, 0, 0))
                screen.blit(x_mark, x_mark.get_rect(center=box_rect.center))
        
        # Display decision message if any
        if decision_message:
            dm = font.render(decision_message, True, (255, 255, 255))
            screen.blit(dm, dm.get_rect(center=(screen_width//2, screen_height//2 - 100)))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    selected_index = (selected_index + 1) % len(options)
                elif event.key == pygame.K_LEFT:
                    selected_index = (selected_index - 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    # Evaluate the selection
                    if options[selected_index]["path"] == correct_icon["path"]:
                        decision_message = "Correct!"
                        pygame.display.flip()
                        pygame.time.delay(1500)
                        selecting = False
                    else:
                        decision_message = "Try Again!"
                        if selected_index not in incorrect_choices:
                            incorrect_choices.append(selected_index)
                        pygame.display.flip()
                        pygame.time.delay(1500)
                        decision_message = ""
        clock.tick(60)
    
    # Final message after tutorial trial
    final_text = "Tutorial complete!\nGet ready for the game!"
    show_message(screen, font, final_text, screen_width, screen_height, clock)

# -------------------------------
# Main Tutorial Setup
# -------------------------------
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Matching Pairs Tutorial")
clock = pygame.time.Clock()
FPS = 60
font = pygame.font.SysFont(None, 50)

# Set your icon directory (adjust the path as needed)
icon_path = "VPAT/assets/vpat/Kids-190-a"

# Load icons and scale them
icons = []
for file in os.listdir(icon_path):
    try:
        icon_img = pygame.image.load(os.path.join(icon_path, file)).convert_alpha()
        icon_img = pygame.transform.scale(icon_img, (70, 70))
        icons.append(icon_img)
    except Exception as e:
        print("Error loading", file, e)
random.shuffle(icons)

# Create one pair from the icons for the tutorial.
pairs = []
num_pairs = 1  # Only one pair is used in the tutorial
for i in range(0, num_pairs * 2, 2):
    if i + 1 < len(icons):
        pair = (
            {"surface": icons[i], "path": os.path.join(icon_path, os.listdir(icon_path)[i])},
            {"surface": icons[i+1], "path": os.path.join(icon_path, os.listdir(icon_path)[i+1])}
        )
        pairs.append(pair)

# Run the tutorial
matching_tutorial(screen, clock, font, SCREEN_WIDTH, SCREEN_HEIGHT, icons, icon_path, pairs)

# Game Functions
def display_pair(pair):
    left_icon = pair[0]["surface"]
    right_icon = pair[1]["surface"]
    left_rect = left_icon.get_rect(center=(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 250))
    right_rect = right_icon.get_rect(center=(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 + 250))
    screen.blit(left_icon, left_rect)
    screen.blit(right_icon, right_rect)

def display_feedback():
    if feedback_message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(feedback_message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 120)))

def display_matching_game(target_icon, random_icons, selected_index, incorrect_choices, animation_state, positions, animation_progress, decision_message):
    # Extract positions
    target_box_pos = positions["target"]
    hovered_box_pos = positions["hovered"]
    animation_speed = 3
    max_distance = 50

    if animation_state == "moving_apart":
        target_box_pos[0] -= animation_speed
        hovered_box_pos[0] += animation_speed
        animation_progress["distance"] += animation_speed
        if animation_progress["distance"] >= max_distance:
            animation_progress["distance"] = 0  
            animation_state = "flying_together"  # Transition to flying together

    elif animation_state == "flying_together":
        target_box_pos[0] += 2 * animation_speed
        hovered_box_pos[0] -= 2 * animation_speed
        if abs(target_box_pos[0] - hovered_box_pos[0]) <= 100:  
            animation_state = "correct"  
            decision_message = "Correct!"

    elif animation_state == "correct" or animation_state == "incorrect":
        # Draw rectangles around clashed squares
        box_color = (0, 255, 0) if animation_state == "correct" else (255, 0, 0)
        pygame.draw.rect(screen, box_color, (*target_box_pos, 100, 100), 5)
        pygame.draw.rect(screen, box_color, (*hovered_box_pos, 100, 100), 5)

    # Draw target and hovered icons
    target_box_rect = pygame.Rect(target_box_pos[0], target_box_pos[1], 100, 100)
    hovered_box_rect = pygame.Rect(hovered_box_pos[0], hovered_box_pos[1], 100, 100)

    pygame.draw.rect(screen, (255, 255, 255), target_box_rect, 3)
    pygame.draw.rect(screen, (255, 255, 255), hovered_box_rect, 3)

    screen.blit(target_icon["surface"], (target_box_pos[0] + 15, target_box_pos[1] + 15))
    screen.blit(random_icons[selected_index]["surface"], (hovered_box_pos[0] + 15, hovered_box_pos[1] + 15))

    # Add "Target" text above the target box
    font = pygame.font.SysFont(None, 50)
    target_text = font.render("Target", True, (0, 255, 0))
    text_rect = target_text.get_rect(center=(target_box_pos[0] + 50, target_box_pos[1] - 50))  # Adjust the position
    screen.blit(target_text, text_rect)
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

        if i == selected_index:
            pygame.draw.rect(screen, (0, 255, 0), box_rect, 3)
        else:
            pygame.draw.rect(screen, (255, 255, 255), box_rect, 3)

        icon_rect = icon["surface"].get_rect(center=box_rect.center)
        screen.blit(icon["surface"], icon_rect)

        if i in incorrect_choices:
            font = pygame.font.SysFont(None, 70)
            text = font.render("X", True, (255, 0, 0))
            screen.blit(text, text.get_rect(center=box_rect.center))

    return animation_state, positions, decision_message

def display_matched_pairs():
    screen_width, screen_height = screen.get_size()
    pair_width = screen_width // 6
    pair_height = screen_height // 10 

    for i, (pair, was_correct) in enumerate(matched_pairs):
        # Determine row and column
        row = i // 5  
        col = i % 5   
        x_offset = (col + 1) * pair_width - 40  
        y_offset = (row + 1) * pair_height  

        left_icon = pygame.transform.scale(pair[0]["surface"], (40, 40))
        right_icon = pygame.transform.scale(pair[1]["surface"], (40, 40))
        left_rect = left_icon.get_rect(topleft=(x_offset, y_offset))
        right_rect = right_icon.get_rect(topleft=(x_offset + 40, y_offset))

        box_color = (0, 0, 255) if was_correct else (255, 0, 0)
        pygame.draw.rect(screen, box_color, left_rect.union(right_rect).inflate(10, 10), 3)
        screen.blit(left_icon, left_rect)
        screen.blit(right_icon, right_rect)

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Load assets
background = pygame.image.load('VPAT/assets/vpat/background.jpg').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
icon_path = 'VPAT/assets/vpat/Kids-190-a'
icons = [pygame.image.load(os.path.join(icon_path, file)).convert_alpha() for file in os.listdir(icon_path)]
icons = [pygame.transform.scale(icon, (70, 70)) for icon in icons]
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
attempts = 0  
feedback_message = ""
feedback_start_time = None
FEEDBACK_DURATION = 1000  
running = True
random_icons = []
incorrect_choices = []
selected_box_index = 0
game_results = []
game_state = "showing_pairs"  # Possible states: 'showing_pairs', 'matching_game'
message = "Remember these pairs!" # Initial message
message_start_time = pygame.time.get_ticks() 
MESSAGE_DURATION = 1200 

animation_state = "idle"  # Possible states: idle, moving_apart, flying_together, correct, incorrect
positions = {
    "target": [SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 50],
    "hovered": [SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT // 2 + 50]
}
animation_progress = {"distance": 0}

decision_message = ""
decision_start_time = None
DECISION_DISPLAY_DURATION = 1500 

while running:
    screen.blit(background, (0, 0))
    display_matched_pairs()

    if game_state == "showing_pairs":
        if pair_index < len(pairs):
            display_pair(pairs[pair_index])
        else:
            game_state = "matching_game"
            message = "Now Match the Pairs!"
            message_start_time = pygame.time.get_ticks() 
            target_icon = pairs[0][0] 
            correct_icon = pairs[0][1] 
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
        animation_state, positions, decision_message = display_matching_game(
            target_icon, random_icons, selected_box_index, incorrect_choices,
            animation_state, positions, animation_progress, decision_message
        )

        if animation_state == "correct" or animation_state == "incorrect":
            if not decision_start_time:
                decision_start_time = pygame.time.get_ticks()

            if pygame.time.get_ticks() - decision_start_time > DECISION_DISPLAY_DURATION:
                decision_start_time = None
                decision_message = ""  
                feedback_message = ""  
                animation_state = "idle" 
                positions["target"] = [SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 50]
                positions["hovered"] = [SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT // 2 + 50]
                animation_progress["distance"] = 0

                if attempts == 0:
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
                    else:
                        feedback_message = "All pairs matched!"
                        running = False

    if pygame.time.get_ticks() - message_start_time < MESSAGE_DURATION and message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 120)))

    if decision_message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(decision_message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 120)))
        if pygame.time.get_ticks() - decision_start_time > DECISION_DISPLAY_DURATION:
            decision_message = ""  
            decision_start_time = None  

    display_feedback()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if game_state == "showing_pairs" and event.key == pygame.K_SPACE:
                pair_index += 1
            elif game_state == "matching_game" and animation_state == "idle":
                if event.key == pygame.K_RIGHT:
                    selected_box_index = (selected_box_index + 1) % len(random_icons)
                elif event.key == pygame.K_LEFT:
                    selected_box_index = (selected_box_index - 1) % len(random_icons)
                elif event.key == pygame.K_DOWN:
                    selected_box_index = (selected_box_index + 5) % len(random_icons)
                elif event.key == pygame.K_UP:
                    selected_box_index = (selected_box_index - 5) % len(random_icons)
                elif event.key == pygame.K_RETURN:
                    if random_icons[selected_box_index]["path"] == correct_icon["path"]:
                        animation_state = "moving_apart" 
                        feedback_message = "Correct!"
                        feedback_start_time = pygame.time.get_ticks()
                        message = "Correct!"
                        matched_pairs.append(((target_icon, correct_icon), True))
                        game_results.append({
                            "prompt": [target_icon["path"], correct_icon["path"]],
                            "number_of_trials": attempts,
                            "correct_or_incorrect": "Correct"
                        })
                        # Prepare the next target or end the game
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
                        feedback_start_time = pygame.time.get_ticks()
                        message = "Try Again!"
                        incorrect_choices.append(selected_box_index)
                        attempts += 1
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
with open("/Data/vpat.json", "w") as f:
    json.dump(game_results, f, indent=4)

pygame.quit()
sys.exit()

pygame.quit()
sys.exit()

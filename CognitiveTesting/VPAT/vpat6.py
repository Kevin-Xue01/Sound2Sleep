import pygame
import sys
import os
import random
import json
import math
from datetime import datetime, timedelta
from vpat_settings import SCREEN_WIDTH, SCREEN_HEIGHT, PHYSICAL_WIDTH, PHYSICAL_HEIGHT

NUM_CHOICES = 3
if NUM_CHOICES < 5:
    NUM_ALLOWED_ATTEMPTS = 2
elif NUM_CHOICES >= 5 and NUM_CHOICES < 7:
    NUM_ALLOWED_ATTEMPTS = 3
elif NUM_CHOICES >= 7 and NUM_CHOICES < 9:
    NUM_ALLOWED_ATTEMPTS = 4
else:
    NUM_ALLOWED_ATTEMPTS = 5

force_correct = False

# Global variable for showing phase timer.
showing_phase_start = None
SHOWING_PHASE_DURATION = 30000  # (For testing; adjust as needed)

# -------------------
# Helper to load pairs from a saved JSON file for a given day offset.
def load_pairs_for_date(days_offset, desired_count):
    folder = "CognitiveTesting/VPAT/Data"
    target_date = datetime.now() - timedelta(days=days_offset)
    date_str = target_date.strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}.json")
    loaded_pairs = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        for entry in data:
            left_path = entry["prompt"][0]
            right_path = entry["prompt"][1]
            try:
                left_img = pygame.image.load(left_path).convert_alpha()
                left_img = pygame.transform.scale(left_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
                right_img = pygame.image.load(right_path).convert_alpha()
                right_img = pygame.transform.scale(right_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
                pair = ({"surface": left_img, "path": left_path},
                        {"surface": right_img, "path": right_path})
                loaded_pairs.append(pair)
            except Exception as e:
                print("Error loading pair:", e)
            if len(loaded_pairs) >= desired_count:
                break
    return loaded_pairs

print(SCREEN_HEIGHT, SCREEN_WIDTH)

# -------------------
# Game Functions – all drawing now uses a passed target_surface
def display_pair(pair, target_surface):
    left_icon = pair[0]["surface"]
    right_icon = pair[1]["surface"]
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    # Offsets are relative fractions of the game width/height.
    offset_x = int(SCREEN_WIDTH * 0.1667)  # roughly SCREEN_WIDTH/6
    offset_y = int(SCREEN_HEIGHT * 0.3333)  # roughly SCREEN_HEIGHT/3
    left_rect = left_icon.get_rect(center=(center_x - offset_x, center_y + offset_y))
    right_rect = right_icon.get_rect(center=(center_x + offset_x, center_y + offset_y))
    target_surface.blit(left_icon, left_rect)
    target_surface.blit(right_icon, right_rect)

def display_feedback(target_surface):
    if feedback_message:
        font_size = int(SCREEN_HEIGHT * 0.05)  # relative font size
        font = pygame.font.SysFont(None, font_size)
        text = font.render(feedback_message, True, (255, 255, 255))
        vertical_offset = int(SCREEN_HEIGHT * 0.1667)
        target_surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - vertical_offset)))

def display_matching_game(target_surface, target_icon, correct_icon, random_icons, selected_index, incorrect_choices, animation_state, positions, animation_progress, decision_message, num_choices):
    # Relative parameters
    animation_speed = max(1, int(SCREEN_WIDTH * 0.00375))   # 3 for 800 width
    max_distance = int(SCREEN_WIDTH * 0.0625)                 # 50 for 800 width

    # Use positions (which are already relative to the game surface)
    target_box_pos = positions["target"]
    hovered_box_pos = positions["hovered"]

    if animation_state == "moving_apart":
        target_box_pos[0] -= animation_speed
        hovered_box_pos[0] += animation_speed
        animation_progress["distance"] += animation_speed
        if animation_progress["distance"] >= max_distance:
            animation_progress["distance"] = 0  
            animation_state = "flying_together"
    elif animation_state == "flying_together":
        target_box_pos[0] += 2 * animation_speed
        hovered_box_pos[0] -= 2 * animation_speed
        if abs(target_box_pos[0] - hovered_box_pos[0]) <= int(SCREEN_WIDTH * 0.125):  # ~100 for 800 width
            animation_state = "correct"
            decision_message = "Correct!"
    elif animation_state in ["correct", "incorrect"]:
        border_width = max(1, int(SCREEN_WIDTH * 0.00625))  # ~5 for 800 width
        box_color = (0, 255, 0) if animation_state == "correct" else (255, 0, 0)
        box_size = int(SCREEN_WIDTH * 0.125)  # ~100 for 800 width
        pygame.draw.rect(target_surface, box_color, (target_box_pos[0], target_box_pos[1], box_size, box_size), border_width)
        pygame.draw.rect(target_surface, box_color, (hovered_box_pos[0], hovered_box_pos[1], box_size, box_size), border_width)

    # Draw white-bordered rectangles
    box_size = int(SCREEN_WIDTH * 0.125)
    border_width_white = max(1, int(SCREEN_WIDTH * 0.00375))  # ~3 for 800 width
    target_box_rect = pygame.Rect(target_box_pos[0], target_box_pos[1], box_size, box_size)
    hovered_box_rect = pygame.Rect(hovered_box_pos[0], hovered_box_pos[1], box_size, box_size)
    pygame.draw.rect(target_surface, (255, 255, 255), target_box_rect, border_width_white)
    pygame.draw.rect(target_surface, (255, 255, 255), hovered_box_rect, border_width_white)
    offset = max(1, int(SCREEN_WIDTH * 0.01875))  # ~15 for 800 width
    target_surface.blit(target_icon["surface"], (target_box_pos[0] + offset, target_box_pos[1] + offset))

    if force_correct:
        target_surface.blit(correct_icon["surface"], (hovered_box_pos[0] + offset, hovered_box_pos[1] + offset))
    else:
        target_surface.blit(random_icons[selected_index]["surface"], (hovered_box_pos[0] + offset, hovered_box_pos[1] + offset))

    # Display "Target" label.
    font_size = int(SCREEN_HEIGHT * 0.05)
    font = pygame.font.SysFont(None, font_size)
    label_offset = int(SCREEN_WIDTH * 0.0625)  # ~50 for 800 width
    target_text = font.render("Target", True, (0, 255, 0))
    text_rect = target_text.get_rect(center=(target_box_pos[0] + label_offset, target_box_pos[1] - label_offset))
    target_surface.blit(target_text, text_rect)

    # Compute grid layout.
    if num_choices == 5:
        grid_cols, grid_rows = 5, 1
    elif num_choices == 10:
        grid_cols, grid_rows = 5, 2
    elif num_choices == 8:
        grid_cols, grid_rows = 4, 2
    elif num_choices == 4:
        grid_cols, grid_rows = 4, 1
    elif num_choices == 6:
        grid_cols, grid_rows = 3, 2
    elif num_choices == 3:
        grid_cols, grid_rows = 3, 1
    else:
        grid_cols = math.ceil(math.sqrt(num_choices))
        grid_rows = math.ceil(num_choices / grid_cols)

    box_width = max(1, int(SCREEN_WIDTH * 0.1))       # ~80 for 800 width
    box_height = max(1, int(SCREEN_WIDTH * 0.1))
    box_spacing = max(1, int(SCREEN_WIDTH * 0.0125))    # ~10 for 800 width
    total_grid_width = grid_cols * box_width + (grid_cols - 1) * box_spacing
    total_grid_height = grid_rows * box_height + (grid_rows - 1) * box_spacing
    grid_start_x = (SCREEN_WIDTH - total_grid_width) // 2
    grid_start_y = SCREEN_HEIGHT - total_grid_height - int(SCREEN_HEIGHT * 0.0833)  # ~50 for 600 height
    for i, icon in enumerate(random_icons):
        row = i // grid_cols
        col = i % grid_cols
        x_pos = grid_start_x + col * (box_width + box_spacing)
        y_pos = grid_start_y + row * (box_height + box_spacing)
        box_rect = pygame.Rect(x_pos, y_pos, box_width, box_height)
        if i == selected_index:
            pygame.draw.rect(target_surface, (0, 255, 0), box_rect, border_width_white)
        else:
            pygame.draw.rect(target_surface, (255, 255, 255), box_rect, border_width_white)
        icon_rect = icon["surface"].get_rect(center=box_rect.center)
        target_surface.blit(icon["surface"], icon_rect)
        if i in incorrect_choices:
            error_font_size = max(1, int(SCREEN_HEIGHT * 0.0875))  # ~70 for 800 height if that works
            error_font = pygame.font.SysFont(None, error_font_size)
            error_text = error_font.render("X", True, (255, 0, 0))
            target_surface.blit(error_text, error_text.get_rect(center=box_rect.center))
    return animation_state, positions, decision_message

def display_matched_pairs(target_surface):
    # Filter out only the correct pairs.
    correct_pairs = [pair for pair in matched_pairs if pair[1] is True]
    
    pair_width = SCREEN_WIDTH // 6
    pair_height = SCREEN_HEIGHT // 10 
    for i, (pair, _) in enumerate(correct_pairs):
        row = i // 5  
        col = i % 5   
        x_offset = (col + 1) * pair_width - int(SCREEN_WIDTH * 0.05)  # relative adjustment
        y_offset = (row + 1) * pair_height  
        icon_size = int(SCREEN_WIDTH * 0.05)  # relative size (~40 for 800 width)
        left_icon = pygame.transform.scale(pair[0]["surface"], (icon_size, icon_size))
        right_icon = pygame.transform.scale(pair[1]["surface"], (icon_size, icon_size))
        left_rect = left_icon.get_rect(topleft=(x_offset, y_offset))
        right_rect = right_icon.get_rect(topleft=(x_offset + icon_size, y_offset))
        border_inflate = int(SCREEN_WIDTH * 0.0125)
        box_color = (0, 0, 255)  # Display correct pairs with blue border
        pygame.draw.rect(target_surface, box_color, left_rect.union(right_rect).inflate(border_inflate, border_inflate), max(1, int(SCREEN_WIDTH * 0.00375)))
        target_surface.blit(left_icon, left_rect)
        target_surface.blit(right_icon, right_rect)

# -------------------
# Initialize Pygame and set up display and surfaces
pygame.init()
screen = pygame.display.set_mode((PHYSICAL_WIDTH, PHYSICAL_HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Create the game surface (drawing area) and compute offsets to center it
game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
offset_x = (PHYSICAL_WIDTH - SCREEN_WIDTH) // 2
offset_y = (PHYSICAL_HEIGHT - SCREEN_HEIGHT) // 2

# Load assets
background = pygame.image.load('CognitiveTesting/VPAT/assets/vpat/background.jpg').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
icon_path = 'CognitiveTesting/VPAT/assets/vpat/Kids-190-a'
icons = [pygame.image.load(os.path.join(icon_path, file)).convert_alpha() for file in os.listdir(icon_path)]
icons = [pygame.transform.scale(icon, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875))) for icon in icons]
random.shuffle(icons)

# -------------------
# Build the pairs list for matching:
today_str = datetime.now().strftime("%Y-%m-%d")
daily_prompts_file = os.path.join("CognitiveTesting","VPAT", "new_prompts.json")

def load_new_pairs_from_daily():
    new_pairs = []
    if os.path.exists(daily_prompts_file):
        with open(daily_prompts_file, "r") as f:
            daily_data = json.load(f)
        for pair in daily_data["prompts"]:
            left_filename = pair["left"]
            right_filename = pair["right"]
            left_path = os.path.join(icon_path, left_filename)
            right_path = os.path.join(icon_path, right_filename)
            try:
                left_img = pygame.image.load(left_path).convert_alpha()
                left_img = pygame.transform.scale(left_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
                right_img = pygame.image.load(right_path).convert_alpha()
                right_img = pygame.transform.scale(right_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
            except Exception as e:
                print("Error loading images from daily prompts:", e)
                continue
            new_pairs.append((
                {"surface": left_img, "path": left_path},
                {"surface": right_img, "path": right_path}
            ))
    return new_pairs

daily_new_pairs = load_new_pairs_from_daily()
if not daily_new_pairs:
    used_pairs_file = os.path.join("CognitiveTesting/VPAT/Pairs", "used_pairs.txt")
    used_images = set()
    if os.path.exists(used_pairs_file):
        with open(used_pairs_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    used_images.update(parts)
    all_files = [file for file in os.listdir(icon_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    available_files = [f for f in all_files if f not in used_images]
    random.shuffle(available_files)
    daily_new_pairs = []
    num_new_pairs = 4
    for i in range(0, len(available_files) - 1, 2):
        if len(daily_new_pairs) >= num_new_pairs:
            break
        left_file = available_files[i]
        right_file = available_files[i+1]
        left_path = os.path.join(icon_path, left_file)
        right_path = os.path.join(icon_path, right_file)
        try:
            left_img = pygame.image.load(left_path).convert_alpha()
            left_img = pygame.transform.scale(left_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
            right_img = pygame.image.load(right_path).convert_alpha()
            right_img = pygame.transform.scale(right_img, (int(SCREEN_WIDTH * 0.0875), int(SCREEN_WIDTH * 0.0875)))
        except Exception as e:
            print("Error loading new pair images:", e)
            continue
        daily_new_pairs.append((
            {"surface": left_img, "path": left_path},
            {"surface": right_img, "path": right_path}
        ))
    daily_data = {
        "day": today_str,
        "prompts": [{"left": pair[0]["path"].split(os.sep)[-1],
                     "right": pair[1]["path"].split(os.sep)[-1]} for pair in daily_new_pairs]
    }
    with open(daily_prompts_file, "w") as f:
        json.dump(daily_data, f, indent=4)
    master_file = os.path.join("CognitiveTesting","VPAT", "master_used_prompts.json")
    if os.path.exists(master_file):
        with open(master_file, "r") as f:
            master_data = json.load(f)
    else:
        master_data = {}
    master_data[today_str] = daily_data["prompts"]
    with open(master_file, "w") as f:
        json.dump(master_data, f, indent=4)

new_pairs = daily_new_pairs

yesterday_pairs = load_pairs_for_date(1, 3)
two_days_pairs = load_pairs_for_date(2, 2)
three_days_pairs = load_pairs_for_date(3, 1)
pairs = new_pairs.copy()
if yesterday_pairs:
    pairs.extend(yesterday_pairs)
if two_days_pairs:
    pairs.extend(two_days_pairs)
if three_days_pairs:
    pairs.extend(three_days_pairs)

# Initialize game variables.
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
game_state = "showing_pairs"  # or "matching_game"
message = "Remember these pairs!"
message_start_time = pygame.time.get_ticks()
MESSAGE_DURATION = 1200
score = 0

animation_state = "idle"
positions = {
    "target": [int(SCREEN_WIDTH * 0.375), int(SCREEN_HEIGHT * 0.5833)],   # e.g., SCREEN_WIDTH//2 - 150 and SCREEN_HEIGHT//2 + 50 for 800x600
    "hovered": [int(SCREEN_WIDTH * 0.625), int(SCREEN_HEIGHT * 0.5833)]
}
animation_progress = {"distance": 0}
decision_message = ""
decision_start_time = None
DECISION_DISPLAY_DURATION = 1500
next_pair = None

# -------------------
# Main Loop
while running:
    current_time = pygame.time.get_ticks()
    # Clear game surface and draw background
    game_surface.blit(background, (0, 0))
    display_matched_pairs(game_surface)

    # --- Showing Phase ---
    if game_state == "showing_pairs":
        if showing_phase_start is None:
            showing_phase_start = current_time
        elapsed = current_time - showing_phase_start
        remaining = max(0, SHOWING_PHASE_DURATION - elapsed) // 1000  # seconds
        display_pair(pairs[pair_index], game_surface)
        font = pygame.font.SysFont(None, int(SCREEN_HEIGHT * 0.05))
        timer_text = font.render(f"Time left: {remaining} s", True, (255, 0, 0))
        game_surface.blit(timer_text, (SCREEN_WIDTH // 2 - int(SCREEN_WIDTH * 0.0625), int(SCREEN_HEIGHT * 0.0833)))

        # ---- Draw navigation arrows ----
        arrow_width = int(SCREEN_WIDTH * 0.05)
        arrow_height = int(SCREEN_HEIGHT * 0.05)
        margin = int(SCREEN_WIDTH * 0.025)
        left_arrow_rect = pygame.Rect(margin, SCREEN_HEIGHT - 2*arrow_height - margin, arrow_width, arrow_height)
        right_arrow_rect = pygame.Rect(SCREEN_WIDTH - arrow_width - margin, SCREEN_HEIGHT - 2*arrow_height - margin, arrow_width, arrow_height)

        # Left arrow (triangle pointing left)
        left_arrow_points = [
            (left_arrow_rect.right, left_arrow_rect.top),
            (left_arrow_rect.left, left_arrow_rect.centery),
            (left_arrow_rect.right, left_arrow_rect.bottom)
        ]
        pygame.draw.polygon(game_surface, (255, 255, 255), left_arrow_points)

        # Right arrow (triangle pointing right)
        right_arrow_points = [
            (right_arrow_rect.left, right_arrow_rect.top),
            (right_arrow_rect.right, right_arrow_rect.centery),
            (right_arrow_rect.left, right_arrow_rect.bottom)
        ]
        pygame.draw.polygon(game_surface, (255, 255, 255), right_arrow_points)

        if elapsed >= SHOWING_PHASE_DURATION:
            game_state = "matching_game"
            showing_phase_start = None
            target_icon = pairs[0][0]
            correct_icon = pairs[0][1]
            available_icons = [
                {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                for icon in icons
                if icon != correct_icon["surface"]
            ]
            random_icons = random.sample(available_icons, NUM_CHOICES - 1) + [correct_icon]
            random.shuffle(random_icons)
            incorrect_choices = []
            attempts = 0

    # --- Matching Phase ---
    elif game_state == "matching_game":
        animation_state, positions, decision_message = display_matching_game(
            game_surface, target_icon, correct_icon, random_icons, selected_box_index, incorrect_choices,
            animation_state, positions, animation_progress, decision_message, num_choices=NUM_CHOICES
        )
        if animation_state in ["correct", "incorrect"]:
            if not decision_start_time:
                decision_start_time = current_time
            if current_time - decision_start_time > DECISION_DISPLAY_DURATION:
                decision_start_time = None
                decision_message = ""
                feedback_message = ""
                animation_state = "idle"
                positions["target"] = [int(SCREEN_WIDTH * 0.375), int(SCREEN_HEIGHT * 0.5833)]
                positions["hovered"] = [int(SCREEN_WIDTH * 0.625), int(SCREEN_HEIGHT * 0.5833)]
                animation_progress["distance"] = 0
                force_correct = False
                if next_pair is not None:
                    target_icon = next_pair[0]
                    correct_icon = next_pair[1]
                    available_icons = [
                        {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                        for icon in icons
                        if icon != correct_icon["surface"]
                    ]
                    random_icons = random.sample(available_icons, NUM_CHOICES - 1) + [correct_icon]
                    random.shuffle(random_icons)
                    incorrect_choices = []
                    attempts = 0
                    next_pair = None
                else:
                    if len(matched_pairs) < len(pairs):
                        target_icon = pairs[len(matched_pairs)][0]
                        correct_icon = pairs[len(matched_pairs)][1]
                        available_icons = [
                            {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                            for icon in icons
                            if icon != correct_icon["surface"]
                        ]
                        random_icons = random.sample(available_icons, NUM_CHOICES - 1) + [correct_icon]
                        random.shuffle(random_icons)
                        incorrect_choices = []
                    else:
                        feedback_message = "All pairs matched!"
                        running = False

    # # --- Display messages ---
    # if current_time - message_start_time < MESSAGE_DURATION and message:
    #     font = pygame.font.SysFont(None, int(SCREEN_HEIGHT * 0.05))
    #     text = font.render(message, True, (255, 255, 255))
    #     game_surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - int(SCREEN_HEIGHT * 0.2))))
    # if decision_message:
    #     font = pygame.font.SysFont(None, int(SCREEN_HEIGHT * 0.05))
    #     text = font.render(decision_message, True, (255, 255, 255))
    #     game_surface.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - int(SCREEN_HEIGHT * 0.2))))
    #     if current_time - decision_start_time > DECISION_DISPLAY_DURATION:
    #         decision_message = ""
    #         decision_start_time = None
    display_feedback(game_surface)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # Convert physical coordinates to game_surface coordinates.
            game_pos = (pos[0] - offset_x, pos[1] - offset_y)
            
            if game_state == "showing_pairs":
                arrow_width = int(SCREEN_WIDTH * 0.1)
                arrow_height = int(SCREEN_HEIGHT * 0.1)
                margin = int(SCREEN_WIDTH * 0.025)
                left_arrow_rect = pygame.Rect(margin, SCREEN_HEIGHT - arrow_height - margin, arrow_width, arrow_height)
                right_arrow_rect = pygame.Rect(SCREEN_WIDTH - arrow_width - margin, SCREEN_HEIGHT - arrow_height - margin, arrow_width, arrow_height)
                
                if left_arrow_rect.collidepoint(game_pos):
                    pair_index = (pair_index - 1) % len(pairs)
                elif right_arrow_rect.collidepoint(game_pos):
                    pair_index = (pair_index + 1) % len(pairs)
            
            # (Rest of your event handling code remains unchanged)

            elif game_state == "matching_game" and animation_state == "idle":
                if NUM_CHOICES == 3:
                    grid_cols, grid_rows = 3, 1
                elif NUM_CHOICES == 5:
                    grid_cols, grid_rows = 5, 1
                elif NUM_CHOICES == 10:
                    grid_cols, grid_rows = 5, 2
                elif NUM_CHOICES == 8:
                    grid_cols, grid_rows = 4, 2
                elif NUM_CHOICES == 4:
                    grid_cols, grid_rows = 4, 1
                elif NUM_CHOICES == 6:
                    grid_cols, grid_rows = 3, 2
                else:
                    grid_cols = math.ceil(math.sqrt(NUM_CHOICES))
                    grid_rows = math.ceil(NUM_CHOICES / grid_cols)
                box_width = max(1, int(SCREEN_WIDTH * 0.1))
                box_height = max(1, int(SCREEN_WIDTH * 0.1))
                box_spacing = max(1, int(SCREEN_WIDTH * 0.0125))
                total_grid_width = grid_cols * box_width + (grid_cols - 1) * box_spacing
                total_grid_height = grid_rows * box_height + (grid_rows - 1) * box_spacing
                grid_start_x = (SCREEN_WIDTH - total_grid_width) // 2
                grid_start_y = SCREEN_HEIGHT - total_grid_height - int(SCREEN_HEIGHT * 0.0833)
                for i in range(len(random_icons)):
                    row = i // grid_cols
                    col = i % grid_cols
                    x_pos = grid_start_x + col * (box_width + box_spacing)
                    y_pos = grid_start_y + row * (box_height + box_spacing)
                    box_rect = pygame.Rect(x_pos, y_pos, box_width, box_height)
                    if box_rect.collidepoint(game_pos):
                        selected_box_index = i
                        if random_icons[selected_box_index]["path"] == correct_icon["path"]:
                            animation_state = "moving_apart"
                            feedback_message = "Correct!"
                            score += NUM_ALLOWED_ATTEMPTS - attempts
                            feedback_start_time = pygame.time.get_ticks()
                            message = "Correct!"
                            matched_pairs.append(((target_icon, correct_icon), True))
                            game_results.append({
                                "prompt": [target_icon["path"], correct_icon["path"]],
                                "number_of_trials": attempts,
                                "correct_or_incorrect": "Correct"
                            })
                            force_correct = True
                            if len(matched_pairs) < len(pairs):
                                next_pair = pairs[len(matched_pairs)]
                            else:
                                feedback_message = "All pairs matched!"
                                running = False
                        else:
                            feedback_message = "Try Again!"
                            feedback_start_time = pygame.time.get_ticks()
                            message = "Try Again!"
                            incorrect_choices.append(selected_box_index)
                            attempts += 1
                            if attempts >= NUM_ALLOWED_ATTEMPTS:
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
                                    random_icons = random.sample(available_icons, NUM_CHOICES - 1) + [correct_icon]
                                    random.shuffle(random_icons)
                                    incorrect_choices = []
                                    attempts = 0
                                else:
                                    feedback_message = "All pairs processed!"
                                    running = False
                        break  # Stop checking boxes.
        # Also allow KEYDOWN events for debugging.
        if event.type == pygame.KEYDOWN:
            if game_state == "showing_pairs":
                if event.key == pygame.K_LEFT:
                    pair_index = (pair_index - 1) % len(pairs)
                elif event.key == pygame.K_RIGHT:
                    pair_index = (pair_index + 1) % len(pairs)
                elif event.key == pygame.K_SPACE:
                    pair_index = (pair_index + 1) % len(pairs)
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
                        score += NUM_ALLOWED_ATTEMPTS - attempts
                        feedback_start_time = pygame.time.get_ticks()
                        message = "Correct!"
                        matched_pairs.append(((target_icon, correct_icon), True))
                        game_results.append({
                            "prompt": [target_icon["path"], correct_icon["path"]],
                            "number_of_trials": attempts,
                            "correct_or_incorrect": "Correct"
                        })
                        force_correct = True
                        if len(matched_pairs) < len(pairs):
                            next_pair = pairs[len(matched_pairs)]
                        else:
                            feedback_message = "All pairs matched!"
                            running = False
                    else:
                        feedback_message = "Try Again!"
                        feedback_start_time = pygame.time.get_ticks()
                        message = "Try Again!"
                        incorrect_choices.append(selected_box_index)
                        attempts += 1
                        if attempts >= NUM_ALLOWED_ATTEMPTS:
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
                                random_icons = random.sample(available_icons, NUM_CHOICES - 1) + [correct_icon]
                                random.shuffle(random_icons)
                                incorrect_choices = []
                                attempts = 0
                            else:
                                feedback_message = "All pairs processed!"
                                running = False

    if feedback_message and pygame.time.get_ticks() - (feedback_start_time or 0) > FEEDBACK_DURATION:
        feedback_message = ""

    # Finally, clear the physical screen and blit the game surface at the computed offsets.
    screen.fill((0, 0, 0))
    screen.blit(game_surface, (offset_x, offset_y))
    pygame.display.flip()
    clock.tick(FPS)

folder = "CognitiveTesting/VPAT/Data"
if not os.path.exists(folder):
    os.makedirs(folder)
current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filenameData = os.path.join(folder, f"VPAT_data_{current_date_time}.json")
with open(filenameData, "w") as f:
    json.dump(game_results, f, indent=4)

folder = "CognitiveTesting/VPAT/Score"

if not os.path.exists(folder):
    os.makedirs(folder)
current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
score = score / len(pairs) * 100
filenameScore = os.path.join(folder, f"VPAT_score_{current_date_time}.json")
with open(filenameScore, "w") as f:
    json.dump(score, f, indent=4)

#Upload Data and Score to Dropbox
upload_to_dropbox(filenameData)  # Upload the encrypted version of trial log file
upload_to_dropbox(filenameScore)  # Upload the encrypted version of score file

pygame.quit()
sys.exit()

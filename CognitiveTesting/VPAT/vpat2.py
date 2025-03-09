import pygame
import sys
import os
import math  # Import math for sine function
import random
import json  # Import JSON library for saving data

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 1000
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
FPS = 60

# Load assets
background = pygame.image.load('assets/sprites/bluegalaxy.png').convert()
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
platform_img = pygame.image.load('assets/vpat/platform.png').convert_alpha()
spaceship_img = pygame.image.load('assets/vpat/spaceship2.png').convert_alpha()
spaceship_img = pygame.transform.scale(spaceship_img, (120, 120))
floor_img = pygame.image.load('assets/vpat/floor.png').convert_alpha()
floor_img = pygame.transform.scale(floor_img, (150, 50))  # Adjust width and height as needed
ufo_img = pygame.image.load('assets/vpat/bad2.png').convert_alpha()
ufo_img = pygame.transform.scale(ufo_img, (350, 350))
blast_img = pygame.image.load('assets/vpat/blast.png').convert_alpha()
blast_img = pygame.transform.scale(blast_img, (70, 70))  # Adjust size as needed

# Icon setup
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
icon_history = pairs[:]

# Game variables
pair_index = 0
pairs_shown = False
round_complete = False
selected_box_index = 0
highlight_color = (0, 255, 0)  # Green for highlighting the selected box

# Feedback settings
feedback_message = ""
feedback_start_time = None
FEEDBACK_DURATION = 1000  # 1 second for feedback display

# UFO bounce settings
ufo_y_base = 180  # Base Y position for the UFO
ufo_bounce_amplitude = 10  # How far it moves up and down
ufo_bounce_speed = 3  # Speed of the bounce
ufo_timer = 0  # A timer to calculate the sine wave for bouncing

# Display functions
def display_pair(pair):
    """Display a pair of icons on the screen."""
    left_icon = pair[0]["surface"]  # Extract the Surface object for the left icon
    right_icon = pair[1]["surface"]  # Extract the Surface object for the right icon

    left_rect = left_icon.get_rect(center=(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 250))
    right_rect = right_icon.get_rect(center=(SCREEN_WIDTH // 2 + 100, SCREEN_HEIGHT // 2 + 250))

    screen.blit(left_icon, left_rect)
    screen.blit(right_icon, right_rect)


def display_feedback():
    """Display feedback text on the screen."""
    if feedback_message:
        font = pygame.font.SysFont(None, 50)
        text = font.render(feedback_message, True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 70)))

def display_matching_game(target_icon, random_icons, selected_index, incorrect_choices):
    """Display the target icon and 10 random icons in a 5x2 grid, marking incorrect choices with a red X."""
    # Display target icon next to the spaceship
    target_rect = target_icon["surface"].get_rect(midleft=(SCREEN_WIDTH // 2 - 35, SCREEN_HEIGHT // 2 + 200))
    screen.blit(target_icon["surface"], target_rect)

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
            pygame.draw.rect(screen, highlight_color, box_rect, 3)
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

def display_message(message):
    """Display a message at the center of the screen."""
    font = pygame.font.SysFont(None, 50)
    text = font.render(message, True, (255, 255, 255))
    screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 150)))

# Game variables
pair_index = 0
task_index = 0  # Tracks the current task
pairs_shown = False
round_complete = False
selected_box_index = 0
highlight_color = (0, 255, 0)  # Green for highlighting the selected box

# Feedback settings
feedback_message = ""
feedback_start_time = None
FEEDBACK_DURATION = 1000  # 1 second for feedback display

# Game loop
running = True
matching_game_active = False
random_icons = []
target_icon = None
incorrect_choices = []  # Track indexes of incorrect choices

# Spaceship bounce settings
spaceship_y_base = SCREEN_HEIGHT // 2 + 50 # Base Y position for the spaceship
spaceship_bounce_amplitude = 10  # How far it moves up and down
spaceship_bounce_speed = 5  # Speed of the bounce
bounce_timer = 0  # A timer to calculate the sine wave for bouncing

# Spaceship shake settings
shake_active = False
shake_timer_start = 0
SHAKE_DURATION = 500  # Shake for 500ms
shake_amplitude = 10  # How far the spaceship moves side to side during a shake
shake_speed = 20  # Speed of the shake

# New variables for spaceship movement
flying_to_corner = True  # Flag to indicate if the spaceship is flying to the corner
spaceship_target_x = 50  # Target X position for the top-left corner
spaceship_target_y = 50  # Target Y position for the top-left corner
spaceship_speed = 5  # Speed of movement to the corner

# Spaceship bounce settings for new idle position
new_spaceship_y_base = spaceship_target_y  # Update base Y position for bouncing after moving

# Additional variables for new messages
show_memorize_message = True
show_guess_message = False
message_start_time = None
MEMORIZE_MESSAGE_DURATION = 2000  # 2 seconds for the "Please Memorize These Pairs" message
GUESS_MESSAGE_DURATION = 2000  # 2 seconds for the "Now Guess the Pairs!" message

# Blast variables
blast_active = False
blast_x, blast_y = -50, 0
blast_speed = 15
spaceship_shake_active = False
ufo_shake_active = False

def reset_blast():
    """Reset the blast to its initial state."""
    global blast_active, blast_x, blast_y
    blast_active = False
    blast_x, blast_y = 0, 0

# Initialize game data
game_data = {
    "pairs": []
}

# Additional variables
attempts = 0  # Track attempts for the current task

while running:
    screen.blit(background, (0, 0))

    if spaceship_shake_active:
        shake_timer_elapsed = pygame.time.get_ticks() - shake_timer_start
        if shake_timer_elapsed > SHAKE_DURATION:
            spaceship_shake_active = False  # Stop spaceship shaking
        else:
            shake_offset = math.sin(shake_timer_elapsed * shake_speed / 100) * shake_amplitude
            spaceship_x_position = SCREEN_WIDTH // 2 + shake_offset
            spaceship_y_position = spaceship_y_base  # Keep vertical position constant during shake
    else:
        # Normal bouncing motion
        bounce_timer += 1 / FPS
        bounce_offset = math.sin(bounce_timer * spaceship_bounce_speed) * spaceship_bounce_amplitude
        spaceship_x_position = SCREEN_WIDTH // 2  # Reset horizontal position
        spaceship_y_position = spaceship_y_base + bounce_offset

    # Draw spaceship
    spaceship_rect = spaceship_img.get_rect(center=(spaceship_x_position, spaceship_y_position))
    screen.blit(spaceship_img, spaceship_rect)

    # Handle UFO motion and shaking
    if matching_game_active:
        if ufo_shake_active:
            shake_timer_elapsed = pygame.time.get_ticks() - shake_timer_start
            if shake_timer_elapsed > SHAKE_DURATION:
                ufo_shake_active = False  # Stop UFO shaking
            else:
                shake_offset = math.sin(shake_timer_elapsed * shake_speed / 100) * shake_amplitude
                ufo_x_position = SCREEN_WIDTH // 2 + shake_offset
        else:
            ufo_x_position = SCREEN_WIDTH // 2  # Reset horizontal position
        ufo_timer += 1 / FPS
        ufo_bounce_offset = math.sin(ufo_timer * ufo_bounce_speed) * ufo_bounce_amplitude
        ufo_y_position = ufo_y_base + ufo_bounce_offset

        # Draw UFO
        ufo_rect = ufo_img.get_rect(center=(ufo_x_position, ufo_y_position))
        screen.blit(ufo_img, ufo_rect)
    
    # Handle blast motion
    if blast_active:
        blast_y -= blast_speed  # Move the blast upward
        blast_rect = blast_img.get_rect(center=(blast_x, blast_y))
        screen.blit(blast_img, blast_rect)

        if blast_y <= ufo_y_position + 30:  # Adjust the threshold as needed
            reset_blast()  # Deactivate the blast
            ufo_shake_active = True  # Activate UFO shake
            shake_timer_start = pygame.time.get_ticks()  # Start shake timer

    # Display "Please Memorize These Pairs" message
    if show_memorize_message:
        display_message("Please Memorize These Pairs")
        if message_start_time is None:
            message_start_time = pygame.time.get_ticks()
        elif pygame.time.get_ticks() - message_start_time > MEMORIZE_MESSAGE_DURATION:
            show_memorize_message = False  # End message
            pairs_shown = False  # Start showing pairs
            message_start_time = None

    # Show pairs
    elif not pairs_shown and not show_guess_message:
        if pair_index < len(pairs):
            display_pair(pairs[pair_index])
        else:
            pairs_shown = True
            show_guess_message = True  # Trigger the "Now Guess the Pairs!" message
            message_start_time = None

    # Display "Now Guess the Pairs!" message
    elif show_guess_message:
        display_message("Now Guess the Pairs!")
        if message_start_time is None:
            message_start_time = pygame.time.get_ticks()
        elif pygame.time.get_ticks() - message_start_time > GUESS_MESSAGE_DURATION:
            show_guess_message = False  # End message
            matching_game_active = True
            task_index = 0  # Start the first task
            target_icon = pairs[task_index][0]  # Use the first icon of the pair as the target (left icon)
            correct_icon = pairs[task_index][1]  # Use the second icon of the pair as the correct answer (right icon)
            # Create random icons with surface and path
            available_icons = [
                {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                for icon in icons
                if icon != correct_icon["surface"]
            ]
            random_icons = random.sample(available_icons, 9) + [correct_icon]
            random.shuffle(random_icons)

            incorrect_choices = []  # Reset incorrect choices for the first task

    # Matching game
    elif matching_game_active:
        display_matching_game(target_icon, random_icons, selected_box_index, incorrect_choices)

    # Feedback
    display_feedback()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if not matching_game_active:
                if event.key == pygame.K_SPACE and not pairs_shown:
                    pair_index += 1
            else:
                if event.key == pygame.K_RIGHT:
                    selected_box_index = (selected_box_index + 1) % len(random_icons)
                elif event.key == pygame.K_LEFT:
                    selected_box_index = (selected_box_index - 1) % len(random_icons)
                elif event.key == pygame.K_DOWN:
                    selected_box_index = (selected_box_index + 5) % len(random_icons)
                elif event.key == pygame.K_UP:
                    selected_box_index = (selected_box_index - 5) % len(random_icons)
                elif event.key == pygame.K_RETURN:
                    # Check if the selected icon matches the correct corresponding icon
                    if random_icons[selected_box_index]["path"] == correct_icon["path"]:
                        feedback_message = "Correct!"
                        attempts += 1
                        game_data["pairs"].append({
                            "pair_index": task_index,
                            "target_icon": target_icon["path"],
                            "correct_icon": correct_icon["path"],
                            "attempts": attempts
                        })
                        task_index += 1  # Move to the next task
                        attempts = 0  # Reset attempts for the next task
                        # Initialize the blast
                        blast_active = True
                        blast_x = spaceship_x_position
                        blast_y = spaceship_y_position - 50  # Start above the spaceship
                        if task_index < len(pairs):  # If tasks remain
                            target_icon = pairs[task_index][0]  # Use the first icon of the next pair as the target
                            correct_icon = pairs[task_index][1]  # Use the second icon of the next pair as the correct answer
                            # Create random icons with surface and path
                            available_icons = [
                                {"surface": icon, "path": os.path.join(icon_path, os.listdir(icon_path)[icons.index(icon)])}
                                for icon in icons
                                if icon != correct_icon["surface"]
                            ]
                            random_icons = random.sample(available_icons, 9) + [correct_icon]
                            random.shuffle(random_icons)
                            incorrect_choices = []  # Reset incorrect choices for the new task
                        else:
                            feedback_message = "All tasks completed!"
                            matching_game_active = False  # End the game
                            running = False  # Exit the main game loop
                            with open("game_results.json", "w") as f:
                                json.dump(game_data, f, indent=4)
                    else:
                        feedback_message = "Incorrect!"
                        incorrect_choices.append(selected_box_index)  # Mark the choice as incorrect
                        attempts += 1  # Increment attempts for the current task
                        spaceship_shake_active = True  # Activate spaceship shake
                        shake_timer_start = pygame.time.get_ticks()  # Start shake timer
                    feedback_start_time = pygame.time.get_ticks()

                    feedback_start_time = pygame.time.get_ticks()
    # Clear feedback after duration
    if feedback_message and pygame.time.get_ticks() - feedback_start_time > FEEDBACK_DURATION:
        feedback_message = ""

    # Update screen and FPS
    pygame.display.flip()
    clock.tick(FPS)
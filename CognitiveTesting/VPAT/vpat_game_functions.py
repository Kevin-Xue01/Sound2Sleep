import pygame
import sys
import os
import random
import json

SCREEN_WIDTH, SCREEN_HEIGHT = 700, 900
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 120)))

def display_matching_game(target_icon, random_icons, selected_index, incorrect_choices, animation_state, positions, animation_progress, decision_message):
    # Extract positions
    target_box_pos = positions["target"]
    hovered_box_pos = positions["hovered"]

    # Set up animation speeds and distances
    animation_speed = 3
    max_distance = 50

    if animation_state == "moving_apart":
        target_box_pos[0] -= animation_speed
        hovered_box_pos[0] += animation_speed
        animation_progress["distance"] += animation_speed
        if animation_progress["distance"] >= max_distance:
            animation_progress["distance"] = 0  # Reset distance
            animation_state = "flying_together"  # Transition to flying together

    elif animation_state == "flying_together":
        target_box_pos[0] += 2 * animation_speed
        hovered_box_pos[0] -= 2 * animation_speed
        if abs(target_box_pos[0] - hovered_box_pos[0]) <= 100:  # Adjust overlap threshold
            animation_state = "correct"  # Transition to correct
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
    # Display icons in grid
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
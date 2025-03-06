import os
import json
import random
from datetime import datetime, timedelta

# Folder where your images are located
icon_folder = "VPAT/assets/vpat/Kids-190-a"
# Only consider common image extensions (adjust as needed)
image_files = [os.path.join(icon_folder, file) 
               for file in os.listdir(icon_folder)
               if file.lower().endswith(('.png', '.bmp', '.jpeg', '.bmp'))]

def generate_game_results(num_pairs):
    """
    Generate a list of game result entries.
    Each entry is a dict with:
      - "prompt": a list of two image paths (a random pair)
      - "number_of_trials": a random integer (simulating the number of attempts)
      - "correct_or_incorrect": randomly "Correct" or "Incorrect"
    """
    results = []
    for _ in range(num_pairs):
        # Pick two distinct images randomly
        pair = random.sample(image_files, 2)
        result = {
            "prompt": pair,
            "number_of_trials": random.randint(1, 5),
            "correct_or_incorrect": random.choice(["Correct", "Incorrect"])
        }
        results.append(result)
    return results

# Define the folder to store the JSON files (simulate VPAT/Data)
data_folder = "VPAT/Data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# We'll simulate three days:
# For example, let's generate files for: day 1 = today - 2, day 2 = today - 1, day 3 = today.
day_offsets = [2, 1, 0]  # Lower offset means more recent
num_new_pairs = 4      # Each day, the game generates 4 new pairs

for offset in day_offsets:
    target_date = datetime.now() - timedelta(days=offset)
    date_str = target_date.strftime("%Y-%m-%d")
    filename = os.path.join(data_folder, f"{date_str}.json")
    data = generate_game_results(num_new_pairs)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Generated sample game results for {date_str} with {len(data)} pairs in {filename}")

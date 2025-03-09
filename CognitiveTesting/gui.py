import os
import sys
import tkinter as tk
import subprocess
import json
import random
import math
from datetime import datetime, timedelta

# Save the original GUI geometry (for the main menu)
ORIGINAL_GEOMETRY = "700x600"

# Create the main Tkinter window (launcher)
root = tk.Tk()
root.title("Game Launcher")
root.geometry(ORIGINAL_GEOMETRY)

# Main menu frame with buttons
menu_frame = tk.Frame(root)
menu_frame.pack(expand=True, fill=tk.BOTH)

def launch_game(script_path):
    # Define the path to the calibration file (adjust folder structure as needed)
    go_nogo_calibration_file = os.path.join("Go-Nogo", "calibration.txt")
    vpat_calibration_file = os.path.join("VPAT", "calibration.txt")
    
    # If calibration file doesn't exist, launch calibration process first.
    if not os.path.exists(go_nogo_calibration_file):
        calibration_script = os.path.abspath(os.path.join("Go-Nogo", "calibration.py"))
        calibration_process = subprocess.Popen([sys.executable, calibration_script])
        calibration_process.wait()  # Wait until calibration completes

    # Hide the main menu while the game is running.
    menu_frame.pack_forget()
    
    # Launch the selected game as a subprocess in its own window.
    process = subprocess.Popen([sys.executable, os.path.abspath(script_path)])
    
    # Periodically check if the game process has finished.
    def check_process():
        if process.poll() is not None:
            # When the game closes, show the main menu again.
            menu_frame.pack(expand=True, fill=tk.BOTH)
        else:
            root.after(100, check_process)
    check_process()

def show_scoreboard():
    # Determine today's date.
    today = datetime.now().strftime("%Y-%m-%d")
    # Assume each game has a Score folder with a file named "YYYY-MM-DD.json".
    score1_file = os.path.join("Go-Nogo", "Score", f"{today}.json")
    score2_file = os.path.join("VPAT", "Score", f"{today}.json")
    
    score1 = 0
    score2 = 0
    if os.path.exists(score1_file):
        try:
            with open(score1_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                score1 = data.get("score", 0)
            elif isinstance(data, (int, float)):
                score1 = data
        except Exception as e:
            print("Error reading score from", score1_file, ":", e)
    if os.path.exists(score2_file):
        try:
            with open(score2_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                score2 = data.get("score", 0)
            elif isinstance(data, (int, float)):
                score2 = data
        except Exception as e:
            print("Error reading score from", score2_file, ":", e)
    
    user_score = score1 + score2

    # Generate five competitor entries.
    names = ["Alice", "Bob", "Charlie", "Diana", "Evan"]
    competitor_scores = []
    # Four random scores less than the user's score.
    for _ in range(4):
        if user_score > 0:
            competitor_scores.append(random.uniform(0, user_score))
        else:
            competitor_scores.append(0)
    # One competitor's score is the user's score multiplied by a random multiplier.
    multipliers = [0.88, 0.99, 0.95, 0.93, 0.97, 0.92, 0.91, 1.05]
    competitor_scores.append(user_score * random.choice(multipliers))
    
    # Combine names and scores, then sort in descending order (highest first).
    competitors = list(zip(names, competitor_scores))
    competitors.sort(key=lambda x: x[1], reverse=True)
    
    # Create a new Tkinter window for the scoreboard.
    scoreboard = tk.Toplevel(root)
    scoreboard.title("Scoreboard")
    scoreboard.geometry("400x300")
    
    label_user = tk.Label(scoreboard, text=f"Your Score: {user_score:.2f}", font=("Arial", 16))
    label_user.pack(pady=10)
    
    # Display competitor entries.
    for name, score in competitors:
        label = tk.Label(scoreboard, text=f"{name}: {score:.2f}", font=("Arial", 14))
        label.pack(pady=2)

# Buttons to launch each game and show the scoreboard.
btn_game1 = tk.Button(
    menu_frame,
    text="Launch Game 1",
    command=lambda: launch_game("./Go-Nogo/game.py"),
    width=20,
    height=2
)
btn_game1.pack(pady=10)

btn_game2 = tk.Button(
    menu_frame,
    text="Launch Game 2",
    command=lambda: launch_game("./VPAT/vpat6.py"),
    width=20,
    height=2
)
btn_game2.pack(pady=10)

btn_scoreboard = tk.Button(
    menu_frame,
    text="Scoreboard",
    command=show_scoreboard,
    width=20,
    height=2
)
btn_scoreboard.pack(pady=10)

# Start the Tkinter event loop.
root.mainloop()

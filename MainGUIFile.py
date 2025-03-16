# Code written by Sean D'Mello (seandmello2002@gmail.com)
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QStackedWidget
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import subprocess
import os
from datetime import datetime
import random
import json

class SleepStudyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overnight Sounds Research Study")
        self.setStyleSheet("background-color: #1A0033;")

        self.stacked_widget = QStackedWidget()
        self.main_page = self.create_main_page()
        self.setup_page = self.create_setup_page()
        self.data_collection_page = self.create_data_collection_page()
        self.mood_page = self.create_mood_selection_page()
        self.morning_sleepiness_page = self.create_morning_sleepiness_test()
        self.game_launch_page = self.create_game_launcher()
        self.scoreboard_page = self.create_scoreboard_page()

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.setup_page)
        self.stacked_widget.addWidget(self.data_collection_page)
        self.stacked_widget.addWidget(self.mood_page)
        self.stacked_widget.addWidget(self.morning_sleepiness_page)
        self.stacked_widget.addWidget(self.game_launch_page)
        self.stacked_widget.addWidget(self.scoreboard_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

    def create_header(self):
        top_layout = QHBoxLayout()

        # Left layout for logo + text
        left_layout = QHBoxLayout()

        logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/seandmello/Downloads/The_Hospital_for_Sick_Children_Logo.svg.png")
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio))

        left_layout.addWidget(logo_label)

        text_layout = QVBoxLayout()
        lab_label = QLabel("Ibrahim Lab")
        lab_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        lab_label.setStyleSheet("color: white;")

        study_label = QLabel("Overnight sounds research study")
        study_label.setFont(QFont("Arial", 10))
        study_label.setStyleSheet("color: white;")

        text_layout.addWidget(lab_label)
        text_layout.addWidget(study_label)
        left_layout.addLayout(text_layout)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        top_layout.addLayout(left_layout)
        top_layout.addStretch()

        status_layout = QVBoxLayout()
        status_label = QLabel("Status:")
        status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        eeg_status = QLabel("\u25CB EEG Paused")
        eeg_status.setFont(QFont("Arial", 12))
        eeg_status.setStyleSheet("color: grey;")

        sounds_status = QLabel("\u25CB Sounds Paused")
        sounds_status.setFont(QFont("Arial", 12))
        sounds_status.setStyleSheet("color: grey;")

        status_layout.addWidget(status_label)
        status_layout.addWidget(eeg_status)
        status_layout.addWidget(sounds_status)
        top_layout.addLayout(status_layout)

        stop_button = QPushButton("Press to STOP")
        stop_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        stop_button.setStyleSheet("background-color: #8B4513; color: white; padding: 10px; border-radius: 10px;")
        top_layout.addWidget(stop_button)

        return top_layout

    def create_main_page(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        layout.addLayout(self.create_header())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Night 1 of 2")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(title_label)

        start_button = QPushButton("Start setup for sleep")
        start_button.setFont(QFont("Arial", 16))
        start_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        start_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.setup_page))
        center_layout.addWidget(start_button)

        sleepiness_button = QPushButton("Do the sleepiness test")
        sleepiness_button.setFont(QFont("Arial", 14))
        sleepiness_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        sleepiness_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.game_launch_page))
        center_layout.addWidget(sleepiness_button)

        layout.addLayout(center_layout)
        main_widget.setLayout(layout)
        return main_widget

    def create_setup_page(self):
        setup_widget = QWidget()
        layout = QVBoxLayout()

        layout.addLayout(self.create_header())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Put on your headband")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(title_label)

        subtitle_label = QLabel("Adjust until you're comfortable and we have a good signal")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(subtitle_label)

        ready_button = QPushButton("I'm ready to sleep!")
        ready_button.setFont(QFont("Arial", 16))
        ready_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        ready_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.data_collection_page))
        center_layout.addWidget(ready_button)

        layout.addLayout(center_layout)
        setup_widget.setLayout(layout)
        return setup_widget

    def create_data_collection_page(self):
        data_widget = QWidget()
        layout = QVBoxLayout()

        layout.addLayout(self.create_header())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Recording Data")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(title_label)

        subtitle_label = QLabel("Good night!")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(subtitle_label)

        done_button = QPushButton("End")
        done_button.setFont(QFont("Arial", 16))
        done_button.setStyleSheet("background-color: #8B0000; color: white; padding: 10px; border-radius: 10px;")
        done_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.mood_page))
        center_layout.addWidget(done_button)

        layout.addLayout(center_layout)
        data_widget.setLayout(layout)
        return data_widget

    def create_mood_selection_page(self):
        mood_widget = QWidget()
        layout = QVBoxLayout()

        layout.addLayout(self.create_header())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        # Title
        good_morning_label = QLabel("Good morning!")
        good_morning_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        good_morning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(good_morning_label)

        question_label = QLabel("How are you feeling right now?")
        question_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(question_label)

        # Mood Buttons
        mood_layout = QHBoxLayout()
        moods = ["😠", "☹️", "😐", "🙂", "😄"]

        for mood in moods:
            button = QPushButton(mood)
            button.setFont(QFont("Arial", 36))
            button.setStyleSheet("background-color: transparent; border: none;")
            button.clicked.connect(lambda _, m=mood: self.stacked_widget.setCurrentWidget(self.morning_sleepiness_page))
            mood_layout.addWidget(button)

        layout.addLayout(mood_layout)
        mood_widget.setLayout(layout)

        return mood_widget

    def create_morning_sleep_test(self):
        mood_widget = QWidget()
        layout = QVBoxLayout()

        layout.addLayout(self.create_header())



        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        # Title
        good_morning_label = QLabel("Good morning!")
        good_morning_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        good_morning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(good_morning_label)

        question_label = QLabel("How are you feeling right now?")
        question_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(question_label)

        # Mood Buttons
        mood_layout = QHBoxLayout()
        moods = ["😠", "☹️", "😐", "🙂", "😄"]

        for mood in moods:
            button = QPushButton(mood)
            button.setFont(QFont("Arial", 36))
            button.setStyleSheet("background-color: transparent; border: none;")
            button.clicked.connect(lambda _, m=mood: self.stacked_widget.setCurrentWidget(self.morning_sleepiness_page))
            mood_layout.addWidget(button)

        layout.addLayout(mood_layout)
        mood_widget.setLayout(layout)

        return mood_widget

    def create_morning_sleepiness_test(self):
        morning_sleepiness_test_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addLayout(self.create_header())

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        # --- Main Content ---
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        good_morning_label = QLabel("Good morning!")
        good_morning_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        good_morning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        good_morning_label.setStyleSheet("color: white;")

        sleep_test_label = QLabel("Let's test how sleepy you are!")
        sleep_test_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        sleep_test_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sleep_test_label.setStyleSheet("color: white;")

        test_button = QPushButton("Do the sleepiness test")
        test_button.setFont(QFont("Arial", 16))
        test_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        test_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.game_launch_page))

        center_layout.addWidget(good_morning_label)
        center_layout.addWidget(sleep_test_label)
        center_layout.addWidget(test_button)

        layout.addLayout(center_layout)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #0c0224;")  # Dark background

        morning_sleepiness_test_widget.setLayout(layout)

        return morning_sleepiness_test_widget

    def create_game_launcher(self):
        game_launcher_widget = QWidget()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(self.create_header())


        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        # --- Main Content ---
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        launch_game1_button = QPushButton("Launch Game 1")
        launch_game1_button.setFont(QFont("Arial", 16))
        launch_game1_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game1_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/Go-Nogo/game.py"))

        launch_game2_button = QPushButton("Launch Game 2")
        launch_game2_button.setFont(QFont("Arial", 16))
        launch_game2_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game2_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/VPAT/vpat6.py"))

        scoreboard_button = QPushButton("Scoreboard")
        scoreboard_button.setFont(QFont("Arial", 16))
        scoreboard_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        scoreboard_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.scoreboard_page))



        center_layout.addWidget(launch_game1_button)
        center_layout.addWidget(launch_game2_button)
        center_layout.addWidget(scoreboard_button)

        # Bottom Buttons (Back and Done)
        bottom_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.setFont(QFont("Arial", 14))
        back_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 5px;")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.main_page))

        done_button = QPushButton("Done")
        done_button.setFont(QFont("Arial", 14))
        done_button.setStyleSheet("background-color: green; color: white; padding: 10px; border-radius: 5px;")

        bottom_layout.addWidget(back_button)
        bottom_layout.addWidget(done_button)

        center_layout.addLayout(bottom_layout)
        layout.addLayout(center_layout)
        self.setStyleSheet("background-color: #0c0224;")  # Dark background

        self.setLayout(layout)
        game_launcher_widget.setLayout(layout)

        return game_launcher_widget

    def launch_game(self, script_path):
        """Launch a game with calibration handling."""
        go_nogo_calibration_file = os.path.join("CognitiveTesting","Go-Nogo", "calibration.txt")

        # Check if calibration is needed
        if script_path.endswith("game.py") and not os.path.exists(go_nogo_calibration_file):
            calibration_script = os.path.abspath(os.path.join("CognitiveTesting", "Go-Nogo", "calibration.py"))
        else:
            calibration_script = None

        if calibration_script:
            calibration_process = subprocess.Popen([sys.executable, calibration_script])
            calibration_process.wait()  # Wait until calibration completes

        # Launch the game
        subprocess.Popen([sys.executable, os.path.abspath(script_path)])


    def create_scoreboard_page(self):
        scoreboard_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addLayout(self.create_header())

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)

        # --- Score Display ---
        today = datetime.now().strftime("%Y-%m-%d")
        score1_file = self.find_first_score_file("CognitiveTesting/Go-Nogo/Score/", "go_no_go_score")
        score2_file = self.find_first_score_file("CognitiveTesting/VPAT/Score/", "vpat_score")

        user_score = self.get_score(score1_file) + self.get_score(score2_file)

        user_label = QLabel(f"Your Score: {user_score:.2f}")
        user_label.setFont(QFont("Arial", 16))
        user_label.setStyleSheet("color: white;")
        layout.addWidget(user_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Competitor Scores ---
        competitors = self.generate_competitor_scores(user_score)
        for name, score in competitors:
            label = QLabel(f"{name}: {score:.2f}")
            label.setFont(QFont("Arial", 14))
            label.setStyleSheet("color: white;")
            layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Back Button ---
        back_button = QPushButton("Back")
        back_button.setFont(QFont("Arial", 14))
        back_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 5px;")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.main_page))
        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet("background-color: #0c0224;")  # Dark background
        scoreboard_widget.setLayout(layout)
        return scoreboard_widget

    def find_first_score_file(self, directory, prefix):
        if not os.path.exists(directory):
            return None
        files = sorted([f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".json")])
        print(files)
        return os.path.join(directory, files[0]) if files else None

    def get_score(self, file_path):
        if not file_path:
            return 0
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                return data.get("score", 0) if isinstance(data, dict) else float(data)
            except Exception as e:
                print(f"Error reading score from {file_path}: {e}")
        return 0


    def generate_competitor_scores(self, user_score):
        names = ["Alice", "Bob", "Charlie", "Diana", "Evan"]
        competitor_scores = [random.uniform(0, user_score) for _ in range(4)] if user_score > 0 else [0] * 4
        competitor_scores.append(user_score * random.choice([0.88, 0.99, 0.95, 0.93, 0.97, 0.92, 0.91, 1.05]))
        competitors = sorted(zip(names, competitor_scores), key=lambda x: x[1], reverse=True)
        return competitors

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepStudyApp()
    window.show()
    sys.exit(app.exec())


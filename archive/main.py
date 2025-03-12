import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QStackedWidget
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import Qt
import subprocess
import os


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

        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.setup_page)
        self.stacked_widget.addWidget(self.data_collection_page)
        self.stacked_widget.addWidget(self.mood_page)
        self.stacked_widget.addWidget(self.morning_sleepiness_page)
        self.stacked_widget.addWidget(self.game_launch_page)

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

        # --- Header ---
        header_layout = QHBoxLayout()

        # Left: SickKids Logo and Study Name
        title_label = QLabel("SickKids\nIbrahim Lab\nOvernight sounds research study")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("color: white;")

        # Right: Status Indicators
        status_layout = QVBoxLayout()
        status_label = QLabel("Status")
        status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status_label.setStyleSheet("color: white;")

        eeg_label = QLabel("○ EEG Paused")
        eeg_label.setFont(QFont("Arial", 12))
        eeg_label.setStyleSheet("color: white;")

        sounds_label = QLabel("○ Sounds Paused")
        sounds_label.setFont(QFont("Arial", 12))
        sounds_label.setStyleSheet("color: white;")

        status_layout.addWidget(status_label)
        status_layout.addWidget(eeg_label)
        status_layout.addWidget(sounds_label)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Stop Button
        stop_button = QPushButton("Press to STOP")
        stop_button.setStyleSheet("background-color: brown; color: white; padding: 10px; border-radius: 5px;")
        stop_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(status_layout)
        header_layout.addWidget(stop_button)

        layout.addLayout(header_layout)

        # --- Separator ---
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

        # --- Header ---
        header_layout = QHBoxLayout()

        title_label = QLabel("SickKids\nIbrahim Lab\nOvernight Sounds Research Study")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("color: white;")

        status_layout = QVBoxLayout()
        status_label = QLabel("Status")
        status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status_label.setStyleSheet("color: white;")

        eeg_label = QLabel("○ EEG Paused")
        eeg_label.setFont(QFont("Arial", 12))
        eeg_label.setStyleSheet("color: white;")

        sounds_label = QLabel("○ Sounds Paused")
        sounds_label.setFont(QFont("Arial", 12))
        sounds_label.setStyleSheet("color: white;")

        status_layout.addWidget(status_label)
        status_layout.addWidget(eeg_label)
        status_layout.addWidget(sounds_label)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        stop_button = QPushButton("Press to STOP")
        stop_button.setStyleSheet("background-color: brown; color: white; padding: 10px; border-radius: 5px;")
        stop_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addLayout(status_layout)
        header_layout.addWidget(stop_button)

        layout.addLayout(header_layout)

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
        launch_game1_button.clicked.connect(lambda: self.launch_game("./Go-Nogo/game.py"))

        launch_game2_button = QPushButton("Launch Game 2")
        launch_game2_button.setFont(QFont("Arial", 16))
        launch_game2_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game2_button.clicked.connect(lambda: self.launch_game("./VPAT/vpat6.py"))

        scoreboard_button = QPushButton("Scoreboard")
        scoreboard_button.setFont(QFont("Arial", 16))
        scoreboard_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        #scoreboard_button.clicked.connect(self.show_scoreboard)

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
        go_nogo_calibration_file = os.path.join("Go-Nogo", "calibration.txt")
        vpat_calibration_file = os.path.join("VPAT", "calibration.txt")

        # Check if calibration is needed
        if script_path.endswith("game.py") and not os.path.exists(go_nogo_calibration_file):
            calibration_script = os.path.abspath(os.path.join("CognitiveTesting", "Go-Nogo", "calibration.py"))
        elif script_path.endswith("vpat6.py") and not os.path.exists(vpat_calibration_file):
            calibration_script = os.path.abspath(os.path.join("CognitiveTesting","VPAT", "calibration.py"))
        else:
            calibration_script = None

        if calibration_script:
            calibration_process = subprocess.Popen([sys.executable, calibration_script])
            calibration_process.wait()  # Wait until calibration completes

        # Launch the game
        subprocess.Popen([sys.executable, os.path.abspath(script_path)])

    # def create_game_launch_page(self):
    #     game_launch_widget = QWidget()
    #     layout = QVBoxLayout()
    #     layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    #
    #     # --- Header ---
    #     header_layout = QHBoxLayout()
    #
    #     # Left: SickKids Logo and Study Name
    #     title_label = QLabel("SickKids\nIbrahim Lab\nOvernight sounds research study")
    #     title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
    #     title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    #     title_label.setStyleSheet("color: white;")
    #
    #     # Right: Status Indicators
    #     status_layout = QVBoxLayout()
    #     status_label = QLabel("Status")
    #     status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    #     status_label.setStyleSheet("color: white;")
    #
    #     eeg_label = QLabel("○ EEG Paused")
    #     eeg_label.setFont(QFont("Arial", 12))
    #     eeg_label.setStyleSheet("color: white;")
    #
    #     sounds_label = QLabel("○ Sounds Paused")
    #     sounds_label.setFont(QFont("Arial", 12))
    #     sounds_label.setStyleSheet("color: white;")
    #
    #     status_layout.addWidget(status_label)
    #     status_layout.addWidget(eeg_label)
    #     status_layout.addWidget(sounds_label)
    #     status_layout.setAlignment(Qt.AlignmentFlag.AlignRight)
    #
    #     # Stop Button
    #     stop_button = QPushButton("Press to STOP")
    #     stop_button.setStyleSheet("background-color: brown; color: white; padding: 10px; border-radius: 5px;")
    #     stop_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    #
    #     header_layout.addWidget(title_label)
    #     header_layout.addStretch()
    #     header_layout.addLayout(status_layout)
    #     header_layout.addWidget(stop_button)
    #
    #     layout.addLayout(header_layout)
    #
    #     # --- Separator ---
    #     separator = QFrame()
    #     separator.setFrameShape(QFrame.Shape.HLine)
    #     separator.setFrameShadow(QFrame.Shadow.Sunken)
    #     separator.setStyleSheet("background-color: white; height: 2px;")
    #     layout.addWidget(separator)
    #
    #     # --- Main Content ---
    #     center_layout = QVBoxLayout()
    #     center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #
    #     launch_game1_button = QPushButton("Launch Game 1")
    #     launch_game1_button.setFont(QFont("Arial", 16))
    #     launch_game1_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
    #
    #     launch_game2_button = QPushButton("Launch Game 2")
    #     launch_game2_button.setFont(QFont("Arial", 16))
    #     launch_game2_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
    #
    #     scoreboard_button = QPushButton("Scoreboard")
    #     scoreboard_button.setFont(QFont("Arial", 16))
    #     scoreboard_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
    #
    #     center_layout.addWidget(launch_game1_button)
    #     center_layout.addWidget(launch_game2_button)
    #     center_layout.addWidget(scoreboard_button)
    #
    #     # Bottom Buttons (Back and Done)
    #     bottom_layout = QHBoxLayout()
    #     back_button = QPushButton("Back")
    #     back_button.setFont(QFont("Arial", 14))
    #     back_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 5px;")
    #     back_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.main_page))
    #
    #     done_button = QPushButton("Done")
    #     done_button.setFont(QFont("Arial", 14))
    #     done_button.setStyleSheet("background-color: green; color: white; padding: 10px; border-radius: 5px;")
    #
    #     bottom_layout.addWidget(back_button)
    #     bottom_layout.addWidget(done_button)
    #
    #     center_layout.addLayout(bottom_layout)
    #     layout.addLayout(center_layout)
    #     self.setStyleSheet("background-color: #0c0224;")  # Dark background
    #
    #     game_launch_widget.setLayout(layout)
    #     return game_launch_widget




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepStudyApp()
    window.show()
    sys.exit(app.exec())


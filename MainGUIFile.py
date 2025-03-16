import sys
import os
import json
import random
import subprocess
from datetime import datetime, timedelta
import io
import matplotlib
matplotlib.use("Agg")  # For headless use, if needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from sleep_staging_functions import generate_sleep_figure, SleepStageReportPage
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QStackedWidget, QSizePolicy, QTextEdit, QScrollArea
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import Qt
from loading_screen import LoadingScreen
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy

# Import the headband connection workflow from your separate file.
from data_collection_gui import HeadbandConnectionWidget

RANKING_DIR = "gui_data/"
if not os.path.exists(RANKING_DIR):
    os.makedirs(RANKING_DIR)

class SleepStudyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Overnight Sounds Research Study")
        self.setStyleSheet("background-color: #1A0033;")
        
        # Launch full screen so that the proportions remain as set.
        self.showFullScreen()

        self.stacked_widget = QStackedWidget()
        self.main_page = self.create_main_page()
        self.data_collection_page = self.create_data_collection_page()
        self.mood_page = self.create_mood_selection_page()
        self.morning_sleepiness_page = self.create_morning_sleepiness_test()
        self.game_launch_page = self.create_game_launcher()
        self.scoreboard_page = self.create_scoreboard_page()

        # Create our new Sleep Stage Report page
        self.sleep_stage_report_page = self.create_sleep_stage_report_page()

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.data_collection_page)
        self.stacked_widget.addWidget(self.mood_page)
        self.stacked_widget.addWidget(self.morning_sleepiness_page)
        self.stacked_widget.addWidget(self.game_launch_page)
        self.stacked_widget.addWidget(self.scoreboard_page)
        self.stacked_widget.addWidget(self.sleep_stage_report_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

    def create_sleep_stage_report_page(self):
        return SleepStageReportPage(self)
    
    def show_sleep_stage_report(self):
        self.show_loading_screen()

    def show_loading_screen(self):
        self.loading_screen = LoadingScreen(self)
        self.stacked_widget.addWidget(self.loading_screen)
        self.stacked_widget.setCurrentWidget(self.loading_screen)
        self.showFullScreen()
        QTimer.singleShot(9600, self.show_sleep_report_after_loading)

    def show_sleep_report_after_loading(self):
        self.stacked_widget.setCurrentWidget(self.sleep_stage_report_page)
        self.sleep_stage_report_page.generate_and_display_report()
        self.stacked_widget.removeWidget(self.loading_screen)

    # --------------------- UI Page Creation Functions --------------------- #

    def create_header(self):
        # Wrap the header layout in a widget so we can fix its height.
        header_widget = QWidget()
        top_layout = QHBoxLayout(header_widget)
        top_layout.setContentsMargins(10, 10, 10, 10)
        # Left side: logo and text
        left_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/seandmello/Downloads/The_Hospital_for_Sick_Children_Logo.svg.png")
        if not logo_pixmap.isNull():
            # Keep the logo size small regardless of full screen.
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
        left_layout.setAlignment(Qt.AlignVCenter)
        top_layout.addLayout(left_layout)
        top_layout.addStretch()
        # Right side: status and stop button
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
        # Fix the header height so it remains small
        header_widget.setFixedHeight(80)
        return header_widget

    def create_main_page(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.create_header())
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Night 1 of 2")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title_label)
        start_button = QPushButton("Start setup for sleep")
        start_button.setFont(QFont("Arial", 16))
        start_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        # Set a maximum width to preserve ratio
        start_button.setMaximumWidth(400)
        start_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.data_collection_page))
        center_layout.addWidget(start_button)
        sleepiness_button = QPushButton("Do the sleepiness test")
        sleepiness_button.setFont(QFont("Arial", 14))
        sleepiness_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        sleepiness_button.setMaximumWidth(400)
        sleepiness_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.game_launch_page))
        center_layout.addWidget(sleepiness_button)
        report_button = QPushButton("View Sleep Stage Report")
        report_button.setFont(QFont("Arial", 14))
        report_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        report_button.setMaximumWidth(400)
        report_button.clicked.connect(self.show_sleep_stage_report)
        center_layout.addWidget(report_button)
        layout.addLayout(center_layout)
        main_widget.setLayout(layout)
        return main_widget

    def create_data_collection_page(self):
        data_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.create_header())
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Put on your headband")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title_label)
        subtitle_label = QLabel("Adjust until you're comfortable!")
        subtitle_label.setFont(QFont("Arial", 16))
        subtitle_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(subtitle_label)
        # Add a new button to start the headband connection (data collection) workflow.
        connect_button = QPushButton("Connect!")
        connect_button.setFont(QFont("Arial", 16))
        connect_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        connect_button.setMaximumWidth(400)
        connect_button.clicked.connect(self.start_headband_connection)
        center_layout.addWidget(connect_button)
        # The "End" button has been removed from this page.
        layout.addLayout(center_layout)
        data_widget.setLayout(layout)
        return data_widget

    def start_headband_connection(self):
        # This method starts the headband connection workflow.
        self.headband_connection = HeadbandConnectionWidget(self)
        self.stacked_widget.addWidget(self.headband_connection)
        # Add an "End" button to the headband connection widget layout.
        # end_button = QPushButton("End")
        # end_button.setFont(QFont("Arial", 16))
        # end_button.setStyleSheet("background-color: #8B0000; color: white; padding: 10px; border-radius: 10px;")
        # end_button.setMaximumWidth(400)
        # end_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.mood_page))
        # # Create a container widget for the button
        # button_container = QWidget()
        # button_layout = QVBoxLayout(button_container)
        # button_layout.setContentsMargins(0, 0, 0, 0)
        # button_layout.addStretch(2)
        # button_layout.addWidget(end_button, alignment=Qt.AlignHCenter)
        # button_layout.addStretch(1)
        # self.headband_connection.main_layout.addWidget(button_container)

        self.stacked_widget.setCurrentWidget(self.headband_connection)

    def create_mood_selection_page(self):
        mood_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.create_header())
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)
        good_morning_label = QLabel("Good morning!")
        good_morning_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        good_morning_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(good_morning_label)
        question_label = QLabel("How are you feeling right now?")
        question_label.setFont(QFont("Arial", 26, QFont.Weight.Bold))
        question_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(question_label)
        mood_layout = QHBoxLayout()
        moods = ["😠", "☹️", "😐", "🙂", "😄"]
        for mood in moods:
            button = QPushButton(mood)
            button.setFont(QFont("Arial", 36))
            button.setStyleSheet("background-color: transparent; border: none;")
            button.setMaximumWidth(150)
            button.clicked.connect(lambda _, m=mood: self.stacked_widget.setCurrentWidget(self.morning_sleepiness_page))
            mood_layout.addWidget(button)
        layout.addLayout(mood_layout)
        mood_widget.setLayout(layout)
        return mood_widget

    def create_morning_sleepiness_test(self):
        morning_sleepiness_test_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(self.create_header())
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        good_morning_label = QLabel("Good morning!")
        good_morning_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        good_morning_label.setAlignment(Qt.AlignCenter)
        good_morning_label.setStyleSheet("color: white;")
        sleep_test_label = QLabel("Let's test how sleepy you are!")
        sleep_test_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        sleep_test_label.setAlignment(Qt.AlignCenter)
        sleep_test_label.setStyleSheet("color: white;")
        test_button = QPushButton("Do the sleepiness test")
        test_button.setFont(QFont("Arial", 16))
        test_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        test_button.setMaximumWidth(400)
        test_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.game_launch_page))
        center_layout.addWidget(good_morning_label)
        center_layout.addWidget(sleep_test_label)
        center_layout.addWidget(test_button)
        layout.addLayout(center_layout)
        self.setStyleSheet("background-color: #0c0224;")
        morning_sleepiness_test_widget.setLayout(layout)
        return morning_sleepiness_test_widget

    def create_game_launcher(self):
        game_launcher_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(self.create_header())
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: white; height: 2px;")
        layout.addWidget(separator)
        center_layout = QVBoxLayout()
        center_layout.setAlignment(Qt.AlignCenter)
        launch_game1_button = QPushButton("Launch Game 1")
        launch_game1_button.setFont(QFont("Arial", 16))
        launch_game1_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game1_button.setMaximumWidth(400)
        launch_game1_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/Go-Nogo/game.py"))
        launch_game2_button = QPushButton("Launch Game 2")
        launch_game2_button.setFont(QFont("Arial", 16))
        launch_game2_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game2_button.setMaximumWidth(400)
        launch_game2_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/VPAT/vpat6.py"))
        scoreboard_button = QPushButton("Scoreboard")
        scoreboard_button.setFont(QFont("Arial", 16))
        scoreboard_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        scoreboard_button.setMaximumWidth(400)
        scoreboard_button.clicked.connect(self.show_scoreboard)
        center_layout.addWidget(launch_game1_button)
        center_layout.addWidget(launch_game2_button)
        center_layout.addWidget(scoreboard_button)
        bottom_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.setFont(QFont("Arial", 14))
        back_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 5px;")
        back_button.setMaximumWidth(200)
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.main_page))
        done_button = QPushButton("Done")
        done_button.setFont(QFont("Arial", 14))
        done_button.setStyleSheet("background-color: green; color: white; padding: 10px; border-radius: 5px;")
        done_button.setMaximumWidth(200)
        bottom_layout.addWidget(back_button)
        bottom_layout.addWidget(done_button)
        center_layout.addLayout(bottom_layout)
        layout.addLayout(center_layout)
        self.setStyleSheet("background-color: #0c0224;")
        game_launcher_widget.setLayout(layout)
        return game_launcher_widget

    def launch_game(self, script_path):
        go_nogo_calibration_file = os.path.join("CognitiveTesting", "Go-Nogo", "calibration.txt")
        if script_path.endswith("game.py") and not os.path.exists(go_nogo_calibration_file):
            calibration_script = os.path.abspath(os.path.join("CognitiveTesting", "Go-Nogo", "calibration.py"))
        else:
            calibration_script = None
        if calibration_script:
            calibration_process = subprocess.Popen([sys.executable, calibration_script])
            calibration_process.wait()
        project_root = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen([sys.executable, os.path.abspath(script_path)], cwd=project_root)

    def show_scoreboard(self):
        self.update_scoreboard()
        self.stacked_widget.setCurrentWidget(self.scoreboard_page)
        self.showFullScreen()

    def create_scoreboard_page(self):
        scoreboard_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        header_label = QLabel("Scoreboard")
        header_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        header_label.setStyleSheet("color: white;")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        self.score_container = QVBoxLayout()
        layout.addLayout(self.score_container)
        back_button = QPushButton("Back to Home")
        back_button.setFont(QFont("Arial", 18))
        back_button.setStyleSheet("background-color: gray; color: white; padding: 10px; border-radius: 5px;")
        back_button.setMaximumWidth(400)
        back_button.clicked.connect(self.exit_fullscreen_and_go_home)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)
        scoreboard_widget.setLayout(layout)
        return scoreboard_widget

    def exit_fullscreen_and_go_home(self):
        self.stacked_widget.setCurrentWidget(self.main_page)
        self.showFullScreen()

    # --------------------------- Ranking Helper Functions --------------------------- #
    def generate_scoreboard_table(self, user_score, competitor_names, forced_competitor):
        entries = [("You", user_score)]

        for name in competitor_names:
            if name == forced_competitor and random.random() < 0.3:
                comp_score = random.uniform(user_score + 0.1, 100)
            else:
                comp_score = random.uniform(0, user_score - 0.1) if user_score > 0.1 else 0
            entries.append((name, comp_score))

        sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)

        ranking_dict = {entry[0]: i + 1 for i, entry in enumerate(sorted_entries)}
        return sorted_entries, ranking_dict

    def load_previous_rankings(self):
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        file_path = os.path.join(RANKING_DIR, f"rankings_{yesterday}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading previous rankings from {file_path}: {e}")
        return {}

    def get_today_rankings(self):
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(RANKING_DIR, f"rankings_{today}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f), file_path
            except Exception as e:
                print(f"Error loading today's rankings from {file_path}: {e}")
        return None, file_path

    def save_today_rankings(self, rankings_dict, file_path):
        try:
            with open(file_path, "w") as f:
                json.dump(rankings_dict, f)
        except Exception as e:
            print(f"Error saving today's rankings to {file_path}: {e}")

    def update_scoreboard(self):
        while self.score_container.count():
            item = self.score_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        main_layout = QVBoxLayout()

        prev_rankings = self.load_previous_rankings()
        today_rankings, ranking_file = self.get_today_rankings()
        if today_rankings is None:
            today_rankings = {}

        ninja_label = QLabel("Ninja Swipe")
        ninja_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        ninja_label.setStyleSheet("color: white;")
        ninja_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(ninja_label)

        competitor_names_ninja = ["James", "Alice", "Bob", "Charlie", "Diana"]
        user_score_ninja = self.get_latest_score("CognitiveTesting/Go-Nogo/Score/")

        if user_score_ninja is None:
            msg = QLabel("Don't Forget to Take the Test!")
            msg.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            msg.setStyleSheet("color: white;")
            msg.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(msg)
        else:
            if "ninja_swipe" in today_rankings:
                scoreboard_data = today_rankings["ninja_swipe"]
                sorted_entries_ninja = [
                    (item[0], item[1]) for item in scoreboard_data["sorted_entries"]
                ]
                ranking_dict_ninja = scoreboard_data["ranking_dict"]
            else:
                sorted_entries_ninja, ranking_dict_ninja = self.generate_scoreboard_table(
                    user_score_ninja,
                    competitor_names_ninja,
                    forced_competitor="Brandon"
                )
                scoreboard_data = {
                    "sorted_entries": [[name, score] for (name, score) in sorted_entries_ninja],
                    "ranking_dict": ranking_dict_ninja
                }
                today_rankings["ninja_swipe"] = scoreboard_data
                self.save_today_rankings(today_rankings, ranking_file)

            grid_ninja = QGridLayout()
            grid_ninja.setHorizontalSpacing(5)
            grid_ninja.setContentsMargins(0, 0, 0, 0)

            header_font = QFont("Arial", 24, QFont.Weight.Bold)
            headers = ["Rank", "Name", "Score", "Ranking Position Change"]
            for col, text in enumerate(headers):
                label = QLabel(text)
                label.setFont(header_font)
                label.setStyleSheet("color: white;")
                label.setAlignment(Qt.AlignCenter)
                grid_ninja.addWidget(label, 0, col)

            for row, (name, score) in enumerate(sorted_entries_ninja, start=1):
                current_rank = ranking_dict_ninja.get(name, row)

                if current_rank == 1:
                    rank_html = ('<div style="display:inline-block; background-color: gold; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">1</div>')
                elif current_rank == 2:
                    rank_html = ('<div style="display:inline-block; background-color: silver; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">2</div>')
                elif current_rank == 3:
                    rank_html = ('<div style="display:inline-block; background-color: #cd7f32; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">3</div>')
                else:
                    rank_html = str(current_rank)

                rank_label = QLabel(rank_html)
                rank_label.setAlignment(Qt.AlignCenter)
                rank_label.setTextFormat(Qt.RichText)
                if current_rank not in [1, 2, 3]:
                    rank_label.setStyleSheet("color:white;")
                grid_ninja.addWidget(rank_label, row, 0)

                name_label = QLabel(name)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setFont(QFont("Arial", 24))
                name_label.setStyleSheet("color:white;")
                grid_ninja.addWidget(name_label, row, 1)

                score_label = QLabel(f"{score:.2f}")
                score_label.setAlignment(Qt.AlignCenter)
                score_label.setFont(QFont("Arial", 24))
                score_label.setStyleSheet("color:white;")
                grid_ninja.addWidget(score_label, row, 2)

                prev_ninja = prev_rankings.get("ninja_swipe", {}).get("ranking_dict", {})
                prev_rank = prev_ninja.get(name, current_rank)
                diff = prev_rank - current_rank
                if diff > 0:
                    change_text = f'<font color="green" style="font-size:24px;">▲{diff}</font>'
                elif diff < 0:
                    change_text = f'<font color="red" style="font-size:24px;">▼{abs(diff)}</font>'
                else:
                    change_text = '<span style="font-size:24px;">-</span>'
                change_label = QLabel(change_text)
                change_label.setAlignment(Qt.AlignCenter)
                change_label.setTextFormat(Qt.RichText)
                grid_ninja.addWidget(change_label, row, 3)

            main_layout.addLayout(grid_ninja)

        spacer = QFrame()
        spacer.setFixedHeight(150)
        spacer.setStyleSheet("background-color: transparent;")
        main_layout.addWidget(spacer)

        memorize_label = QLabel("Can you Memorize?")
        memorize_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        memorize_label.setStyleSheet("color: white;")
        memorize_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(memorize_label)

        competitor_names_memorize = ["James", "Alice", "Bob", "Charlie", "Diana"]
        user_score_memorize = self.get_latest_score("CognitiveTesting/VPAT/Score/")

        if user_score_memorize is None:
            msg = QLabel("Don't Forget to Take the Test!")
            msg.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            msg.setStyleSheet("color: white;")
            msg.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(msg)
        else:
            if "can_you_memorize" in today_rankings:
                scoreboard_data = today_rankings["can_you_memorize"]
                sorted_entries_memorize = [
                    (item[0], item[1]) for item in scoreboard_data["sorted_entries"]
                ]
                ranking_dict_memorize = scoreboard_data["ranking_dict"]
            else:
                sorted_entries_memorize, ranking_dict_memorize = self.generate_scoreboard_table(
                    user_score_memorize,
                    competitor_names_memorize,
                    forced_competitor="James"
                )
                scoreboard_data = {
                    "sorted_entries": [[name, score] for (name, score) in sorted_entries_memorize],
                    "ranking_dict": ranking_dict_memorize
                }
                today_rankings["can_you_memorize"] = scoreboard_data
                self.save_today_rankings(today_rankings, ranking_file)
            grid_memorize = QGridLayout()
            grid_memorize.setHorizontalSpacing(5)
            grid_memorize.setContentsMargins(0, 0, 0, 0)

            for col, text in enumerate(headers):
                label = QLabel(text)
                label.setFont(header_font)
                label.setStyleSheet("color: white;")
                label.setAlignment(Qt.AlignCenter)
                grid_memorize.addWidget(label, 0, col)

            for row, (name, score) in enumerate(sorted_entries_memorize, start=1):
                current_rank = ranking_dict_memorize.get(name, row)

                if current_rank == 1:
                    rank_html = ('<div style="display:inline-block; background-color: gold; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">1</div>')
                elif current_rank == 2:
                    rank_html = ('<div style="display:inline-block; background-color: silver; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">2</div>')
                elif current_rank == 3:
                    rank_html = ('<div style="display:inline-block; background-color: #cd7f32; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">3</div>')
                else:
                    rank_html = str(current_rank)

                rank_label = QLabel(rank_html)
                rank_label.setAlignment(Qt.AlignCenter)
                rank_label.setTextFormat(Qt.RichText)
                if current_rank not in [1, 2, 3]:
                    rank_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(rank_label, row, 0)

                name_label = QLabel(name)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setFont(QFont("Arial", 24))
                name_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(name_label, row, 1)

                score_label = QLabel(f"{score:.2f}")
                score_label.setAlignment(Qt.AlignCenter)
                score_label.setFont(QFont("Arial", 24))
                score_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(score_label, row, 2)

                prev_memorize = prev_rankings.get("can_you_memorize", {}).get("ranking_dict", {})
                prev_rank = prev_memorize.get(name, current_rank)
                diff = prev_rank - current_rank
                if diff > 0:
                    change_text = f'<font color="green" style="font-size:24px;">▲{diff}</font>'
                elif diff < 0:
                    change_text = f'<font color="red" style="font-size:24px;">▼{abs(diff)}</font>'
                else:
                    change_text = '<span style="font-size:24px;">-</span>'
                change_label = QLabel(change_text)
                change_label.setAlignment(Qt.AlignCenter)
                change_label.setTextFormat(Qt.RichText)
                grid_memorize.addWidget(change_label, row, 3)

            main_layout.addLayout(grid_memorize)

        spacer = QFrame()
        spacer.setFixedHeight(150)
        spacer.setStyleSheet("background-color: transparent;")
        main_layout.addWidget(spacer)

        memorize_label = QLabel("Can you Memorize?")
        memorize_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        memorize_label.setStyleSheet("color: white;")
        memorize_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(memorize_label)

        competitor_names_memorize = ["James", "Alice", "Bob", "Charlie", "Diana"]
        user_score_memorize = self.get_latest_score("CognitiveTesting/VPAT/Score/")

        if user_score_memorize is None:
            msg = QLabel("Don't Forget to Take the Test!")
            msg.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            msg.setStyleSheet("color: white;")
            msg.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(msg)
        else:
            if "can_you_memorize" in today_rankings:
                scoreboard_data = today_rankings["can_you_memorize"]
                sorted_entries_memorize = [
                    (item[0], item[1]) for item in scoreboard_data["sorted_entries"]
                ]
                ranking_dict_memorize = scoreboard_data["ranking_dict"]
            else:
                sorted_entries_memorize, ranking_dict_memorize = self.generate_scoreboard_table(
                    user_score_memorize,
                    competitor_names_memorize,
                    forced_competitor="James"
                )
                scoreboard_data = {
                    "sorted_entries": [[name, score] for (name, score) in sorted_entries_memorize],
                    "ranking_dict": ranking_dict_memorize
                }
                today_rankings["can_you_memorize"] = scoreboard_data
                self.save_today_rankings(today_rankings, ranking_file)
            grid_memorize = QGridLayout()
            grid_memorize.setHorizontalSpacing(5)
            grid_memorize.setContentsMargins(0, 0, 0, 0)

            for col, text in enumerate(headers):
                label = QLabel(text)
                label.setFont(header_font)
                label.setStyleSheet("color: white;")
                label.setAlignment(Qt.AlignCenter)
                grid_memorize.addWidget(label, 0, col)

            for row, (name, score) in enumerate(sorted_entries_memorize, start=1):
                current_rank = ranking_dict_memorize.get(name, row)

                if current_rank == 1:
                    rank_html = ('<div style="display:inline-block; background-color: gold; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">1</div>')
                elif current_rank == 2:
                    rank_html = ('<div style="display:inline-block; background-color: silver; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">2</div>')
                elif current_rank == 3:
                    rank_html = ('<div style="display:inline-block; background-color: #cd7f32; border-radius:50%; '
                                'width:35px; height:35px; text-align:center; line-height:35px; color:black; '
                                'font-weight:bold; font-size:22px;">3</div>')
                else:
                    rank_html = str(current_rank)

                rank_label = QLabel(rank_html)
                rank_label.setAlignment(Qt.AlignCenter)
                rank_label.setTextFormat(Qt.RichText)
                if current_rank not in [1, 2, 3]:
                    rank_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(rank_label, row, 0)

                name_label = QLabel(name)
                name_label.setAlignment(Qt.AlignCenter)
                name_label.setFont(QFont("Arial", 24))
                name_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(name_label, row, 1)

                score_label = QLabel(f"{score:.2f}")
                score_label.setAlignment(Qt.AlignCenter)
                score_label.setFont(QFont("Arial", 24))
                score_label.setStyleSheet("color:white;")
                grid_memorize.addWidget(score_label, row, 2)

                prev_memorize = prev_rankings.get("can_you_memorize", {}).get("ranking_dict", {})
                prev_rank = prev_memorize.get(name, current_rank)
                diff = prev_rank - current_rank
                if diff > 0:
                    change_text = f'<font color="green" style="font-size:24px;">▲{diff}</font>'
                elif diff < 0:
                    change_text = f'<font color="red" style="font-size:24px;">▼{abs(diff)}</font>'
                else:
                    change_text = '<span style="font-size:24px;">-</span>'
                change_label = QLabel(change_text)
                change_label.setAlignment(Qt.AlignCenter)
                change_label.setTextFormat(Qt.RichText)
                grid_memorize.addWidget(change_label, row, 3)

            main_layout.addLayout(grid_memorize)

        self.save_today_rankings(today_rankings, ranking_file)

        self.score_container.addLayout(main_layout)

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

    def get_latest_score(self, directory):
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        file_today = os.path.join(directory, f"{today}.json")
        file_yesterday = os.path.join(directory, f"{yesterday}.json")
        if os.path.exists(file_today):
            return self.get_score(file_today)
        elif os.path.exists(file_yesterday):
            return self.get_score(file_yesterday)
        else:
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepStudyApp()
    window.show()
    sys.exit(app.exec())

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QStackedWidget
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import subprocess
import os
from datetime import datetime, timedelta
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

    # --------------------- UI Page Creation Functions --------------------- #

    def create_header(self):
        top_layout = QHBoxLayout()
        # Left side: logo and text
        left_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_pixmap = QPixmap("/Users/seandmello/Downloads/The_Hospital_for_Sick_Children_Logo.svg.png")
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaled(50,50,Qt.AspectRatioMode.KeepAspectRatio))
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
        center_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Night 1 of 2")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
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
        center_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Put on your headband")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title_label)
        subtitle_label = QLabel("Adjust until you're comfortable and we have a good signal")
        subtitle_label.setFont(QFont("Arial", 16))
        subtitle_label.setAlignment(Qt.AlignCenter)
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
        center_layout.setAlignment(Qt.AlignCenter)
        title_label = QLabel("Recording Data")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(title_label)
        subtitle_label = QLabel("Good night!")
        subtitle_label.setFont(QFont("Arial", 16))
        subtitle_label.setAlignment(Qt.AlignCenter)
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
            button.clicked.connect(lambda _, m=mood: self.stacked_widget.setCurrentWidget(self.morning_sleepiness_page))
            mood_layout.addWidget(button)
        layout.addLayout(mood_layout)
        mood_widget.setLayout(layout)
        return mood_widget

    def create_morning_sleepiness_test(self):
        morning_sleepiness_test_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addLayout(self.create_header())
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
        layout.addLayout(self.create_header())
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
        launch_game1_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/Go-Nogo/game.py"))
        launch_game2_button = QPushButton("Launch Game 2")
        launch_game2_button.setFont(QFont("Arial", 16))
        launch_game2_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        launch_game2_button.clicked.connect(lambda: self.launch_game("CognitiveTesting/VPAT/vpat6.py"))
        scoreboard_button = QPushButton("Scoreboard")
        scoreboard_button.setFont(QFont("Arial", 16))
        scoreboard_button.setStyleSheet("background-color: purple; color: white; padding: 10px; border-radius: 10px;")
        scoreboard_button.clicked.connect(self.show_scoreboard)
        center_layout.addWidget(launch_game1_button)
        center_layout.addWidget(launch_game2_button)
        center_layout.addWidget(scoreboard_button)
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
        self.setStyleSheet("background-color: #0c0224;")
        game_launcher_widget.setLayout(layout)
        return game_launcher_widget

    def launch_game(self, script_path):
        """Launch a game with calibration handling."""
        go_nogo_calibration_file = os.path.join("CognitiveTesting", "Go-Nogo", "calibration.txt")
        if script_path.endswith("game.py") and not os.path.exists(go_nogo_calibration_file):
            calibration_script = os.path.abspath(os.path.join("CognitiveTesting", "Go-Nogo", "calibration.py"))
        else:
            calibration_script = None
        if calibration_script:
            calibration_process = subprocess.Popen([sys.executable, calibration_script])
            calibration_process.wait()
        subprocess.Popen([sys.executable, os.path.abspath(script_path)])

    def show_scoreboard(self):
        """Update scoreboard content, show the scoreboard page in full screen."""
        self.update_scoreboard()
        self.stacked_widget.setCurrentWidget(self.scoreboard_page)
        self.showFullScreen()

    def create_scoreboard_page(self):
        """Creates the scoreboard page with a header and container for our two tables."""
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
        back_button.clicked.connect(self.exit_fullscreen_and_go_home)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)
        scoreboard_widget.setLayout(layout)
        return scoreboard_widget

    def exit_fullscreen_and_go_home(self):
        self.showNormal()
        self.stacked_widget.setCurrentWidget(self.main_page)

    # --------------------------- Ranking Helper Functions --------------------------- #
    def generate_scoreboard_table(self, user_score, competitor_names, forced_competitor):
        """
        Generates a scoreboard table for one game.
        Returns a sorted list of entries (tuples of (name, score)) and a ranking dictionary mapping name to rank (1-indexed).
        The user ("You") is placed among 6 total entries (user + 5 competitors) based on weighted probabilities.
        With 30% chance the forced_competitor is placed above the user.
        """
        total_entries = 6
        r = random.random()
        if r < 1/3:
            user_rank = 0
        elif r < 2/3:
            user_rank = 1
        elif r < 2/3 + 1/6:
            user_rank = 2
        else:
            user_rank = 3
        force_flag = False
        if forced_competitor in competitor_names and random.random() < 0.3:
            force_flag = True
            if user_rank == 0:
                user_rank = 1
        score_by_rank = {}
        score_by_rank[user_rank] = user_score
        current = user_score
        for pos in range(user_rank - 1, -1, -1):
            current += random.uniform(1, 10)
            score_by_rank[pos] = current
        current = user_score
        for pos in range(user_rank + 1, total_entries):
            current -= random.uniform(1, 10)
            score_by_rank[pos] = current
        entries = {}
        entries[user_rank] = ("You", user_score)
        available_positions = [pos for pos in range(total_entries) if pos != user_rank]
        names_copy = competitor_names.copy()
        if force_flag:
            possible_positions = [p for p in available_positions if p < user_rank]
            if possible_positions:
                best_pos = min(possible_positions)
                entries[best_pos] = (forced_competitor, score_by_rank[best_pos])
                names_copy.remove(forced_competitor)
                available_positions.remove(best_pos)
        random.shuffle(available_positions)
        for name in names_copy:
            if available_positions:
                pos = available_positions.pop(0)
                entries[pos] = (name, score_by_rank[pos])
        sorted_entries = []
        ranking_dict = {}
        for pos in sorted(entries.keys()):
            entry = entries[pos]
            sorted_entries.append(entry)
            ranking_dict[entry[0]] = pos + 1
        return sorted_entries, ranking_dict

    def load_previous_rankings(self):
        """Load yesterday's ranking file if available, otherwise return an empty dictionary."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        file_path = f"rankings_{yesterday}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading previous rankings from {file_path}: {e}")
        return {}

    def get_today_rankings(self):
        """Load today's ranking file if it exists; otherwise return None and the expected file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = f"rankings_{today}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f), file_path
            except Exception as e:
                print(f"Error loading today's rankings from {file_path}: {e}")
        return None, file_path

    def save_today_rankings(self, rankings_dict, file_path):
        """Save the given rankings dictionary to the specified file path."""
        try:
            with open(file_path, "w") as f:
                json.dump(rankings_dict, f)
        except Exception as e:
            print(f"Error saving today's rankings to {file_path}: {e}")

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
        """Check for today's score file; if missing, use yesterday's; if neither exists, return None."""
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

    def update_scoreboard(self):
        """Rebuilds the scoreboard page with two tables – one for Ninja Swipe (Go-No-Go) and one for Can you Memorize (VPAT) – 
        using yesterday's rankings for change comparison and today's rankings for current display.
        Once today's rankings are generated, they are saved so they remain constant throughout the day."""
        while self.score_container.count():
            item = self.score_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        main_layout = QVBoxLayout()
        prev_rankings = self.load_previous_rankings()  # keys: "ninja_swipe" and "can_you_memorize"
        today_rankings, ranking_file = self.get_today_rankings()
        if today_rankings is None:
            today_rankings = {}

        # ----- Ninja Swipe Table (Go-No-Go) -----
        ninja_label = QLabel("Ninja Swipe")
        ninja_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        ninja_label.setStyleSheet("color: white;")
        ninja_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(ninja_label)

        competitor_names_ninja = ["Brandon", "Olivia", "Jamal", "Ray", "Diana"]
        user_score_ninja = self.get_latest_score("CognitiveTesting/Go-Nogo/Score/")
        if user_score_ninja is None:
            msg = QLabel("Don't Forget to Take the Test!")
            msg.setFont(QFont("Arial", 20, QFont.Weight.Bold))
            msg.setStyleSheet("color: white;")
            msg.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(msg)
        else:
            if "ninja_swipe" in today_rankings:
                ranking_dict_ninja = today_rankings["ninja_swipe"]
                sorted_entries_ninja, _ = self.generate_scoreboard_table(user_score_ninja, competitor_names_ninja, forced_competitor="Brandon")
                sorted_entries_ninja.sort(key=lambda entry: ranking_dict_ninja.get(entry[0], 9999))
            else:
                sorted_entries_ninja, ranking_dict_ninja = self.generate_scoreboard_table(user_score_ninja, competitor_names_ninja, forced_competitor="Brandon")
                today_rankings["ninja_swipe"] = ranking_dict_ninja
                self.save_today_rankings(today_rankings, ranking_file)

            grid_ninja = QGridLayout()
            grid_ninja.setHorizontalSpacing(5)
            grid_ninja.setContentsMargins(0,0,0,0)
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
                if current_rank not in [1,2,3]:
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

                prev_ninja = prev_rankings.get("ninja_swipe", {})
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

        # ----- Can you Memorize Table (VPAT) -----
        memorize_label = QLabel("Can you Memorize")
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
                ranking_dict_memorize = today_rankings["can_you_memorize"]
                sorted_entries_memorize, _ = self.generate_scoreboard_table(user_score_memorize, competitor_names_memorize, forced_competitor="James")
                sorted_entries_memorize.sort(key=lambda entry: ranking_dict_memorize.get(entry[0], 9999))
            else:
                sorted_entries_memorize, ranking_dict_memorize = self.generate_scoreboard_table(user_score_memorize, competitor_names_memorize, forced_competitor="James")
                today_rankings["can_you_memorize"] = ranking_dict_memorize
                self.save_today_rankings(today_rankings, ranking_file)

            grid_memorize = QGridLayout()
            grid_memorize.setHorizontalSpacing(5)
            grid_memorize.setContentsMargins(0,0,0,0)
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
                if current_rank not in [1,2,3]:
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

                prev_memorize = prev_rankings.get("can_you_memorize", {})
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

    # --------------------------- File Handling Functions --------------------------- #
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
        """Check for today's score file; if missing, use yesterday's; if neither exists, return None."""
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

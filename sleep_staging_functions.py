import io
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import yasa
from matplotlib.patches import Arc, FancyArrowPatch, Patch, Rectangle
from mne.datasets.sleep_physionet.age import fetch_data
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import convolve1d

from utils import FileReader, SessionConfig


def hypnogramCreation(file_reader: FileReader, chosen_channel="Ch2", num_channels=4, sfreq=256, epoch_length=30):
    eeg_data = []
    for curr_eeg_data, curr_timestamps in file_reader.read_all_frames():
        eeg_data.append(curr_eeg_data.transpose())
    eeg_data = np.array(eeg_data).reshape(num_channels, -1)

    # Create MNE Info object
    ch_names = [f"Ch{i + 1}" for i in range(num_channels)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")

    # Create MNE RawArray object
    raw = mne.io.RawArray(eeg_data, info)

    # Apply a bandpass filter (optional but recommended for sleep staging)
    raw.filter(0.3, 40., fir_design="firwin")

    # Predict sleep stages with YASA
    sls = yasa.SleepStaging(raw, eeg_name=chosen_channel)  # Assuming Ch1 is a reliable sleep channel
    hypnogram = sls.predict()

    # Convert hypnogram to YASA format (seconds-based)
    hypnogram_sec = yasa.hypno_str_to_int(hypnogram)

    return hypnogram_sec

def add_group_arc(ax, theta_start, theta_end, group_label, arrow_color, r_arrow=1.05, is_ls=False, is_deep=False):
    arc = Arc((0, 0), width=2*r_arrow, height=2*r_arrow,
              angle=0, theta1=theta_start, theta2=theta_end, lw=2, color=arrow_color)
    ax.add_patch(arc)

    mid_angle = (theta_start + theta_end) / 2.0
    mid_rad = np.deg2rad(mid_angle)
    if is_ls:
        x_text = (r_arrow + 0.4) * np.cos(mid_rad)
        y_text = (r_arrow + 0.4) * np.sin(mid_rad)
    elif is_deep:
        x_text = (r_arrow + 0.15) * np.cos(mid_rad)
        y_text = (r_arrow + 0.15) * np.sin(mid_rad)
    else:
        x_text = (r_arrow + 0.3) * np.cos(mid_rad)
        y_text = (r_arrow + 0.3) * np.sin(mid_rad)
    ax.text(x_text, y_text, group_label, ha='center', va='center',
            fontsize=12, color='white', fontweight='bold')

    # Add an arrow at the end of the arc
    end_rad = np.deg2rad(theta_end)
    x_end = r_arrow * np.cos(end_rad)
    y_end = r_arrow * np.sin(end_rad)
    dx = 0.1 * (-np.sin(end_rad))
    dy = 0.1 * (np.cos(end_rad))
    arrow = FancyArrowPatch((x_end, y_end), (x_end + dx, y_end + dy),
                            arrowstyle='-|>', mutation_scale=12,
                            color=arrow_color, lw=2, shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)

def generate_sleep_figure(file_reader):
    hypno_pred = hypnogramCreation(file_reader, "Ch2")

    # Prepare donut chart data
    stage_names = ["Awake", "REM", "N1", "N2", "N3"]
    counts = {name: 0 for name in stage_names}
    for s in hypno_pred:
        if s == 0:
            counts["Awake"] += 1
        elif s == 4:
            counts["REM"] += 1
        elif s == 1:
            counts["N1"] += 1
        elif s == 2:
            counts["N2"] += 1
        elif s == 3:
            counts["N3"] += 1

    data = [counts[name] for name in stage_names]
    donut_colors = ["#FFC179", "#F195AC", "#E3F195", "#95F1A8", "#95C8F1"]

    def format_time(count):
        total_seconds = count * 30
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

    sleep_count = counts["REM"] + counts["N1"] + counts["N2"] + counts["N3"]
    total_sleep_str = format_time(sleep_count)

    if sleep_count > 0:
        TST = sleep_count * 0.5  # total sleep time in minutes
        sleep_duration_factor = min(1, TST / 420)  # 7 hours = 420 minutes
        rem_pct = counts["REM"] / sleep_count
        n1_pct  = counts["N1"] / sleep_count
        n2_pct  = counts["N2"] / sleep_count
        n3_pct  = counts["N3"] / sleep_count
        sleep_quality_factor = 0.35 * n3_pct + 0.25 * rem_pct + 0.40 * (n1_pct + n2_pct)
        sleep_score = int((0.5 * sleep_duration_factor + 0.5 * sleep_quality_factor) * 100)
    else:
        sleep_score = 0

    stage_map = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    hypno_mapped = np.array([stage_map[s] for s in hypno_pred])
    epochs = np.arange(len(hypno_mapped))

    hypno_colors = {
        4: "#FFC179",  # Awake
        3: "#F195AC",  # REM
        2: "#E3F195",  # N1
        1: "#95F1A8",  # N2
        0: "#95C8F1",  # N3
    }

    # Get the current screen size using QApplication (assumes a QApplication is running)
    screen = QApplication.primaryScreen()
    size = screen.size()
    screen_width = size.width()
    screen_height = size.height()

    # Define a DPI and compute a figure size (in inches) that is 80% of the screen dimensions
    dpi = 100
    fig_width = (screen_width / dpi) * 0.8
    fig_height = (screen_height / dpi) * 0.8

    # Create the figure and subplots using the computed size
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("#1A0033")
    ax0.set_facecolor("#1A0033")
    ax1.set_facecolor("#1A0033")

    # --- Donut Chart (ax0) ---
    width = 0.3
    wedges, _ = ax0.pie(
        data, colors=donut_colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=width, edgecolor='white')
    )
    ax0.axis('equal')
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_frame_on(False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # Move the Sleep Score text above the hypnogram by increasing the y coordinate.
    ax0.text(0.5, 1.15, f"Sleep Score: {sleep_score}", ha='center', va='bottom',
             fontsize=16, fontweight='bold', color='white', transform=ax0.transAxes)
    
    for wedge, label_name in zip(wedges, stage_names):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        rad = np.deg2rad(angle)
        r = 1 - (width / 2)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        ax0.text(x, y, label_name, ha='center', va='center',
                 fontsize=10, color='black', fontweight='bold')

    stage_lines = [f"{stg}: {format_time(counts[stg])}" for stg in stage_names]
    stage_text = "\n".join(stage_lines)
    ax0.text(0.5, -0.15, stage_text, ha='center', va='top', fontsize=11,
             color='white', transform=ax0.transAxes)

    ax0.text(0, 0, f"Total Sleep Time:\n{total_sleep_str}", 
             ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    gap_deg = 5 

    wedge_REM = wedges[1]  
    theta_rem_start = wedge_REM.theta1 + gap_deg/2
    theta_rem_end   = wedge_REM.theta2 - 2*gap_deg/2
    add_group_arc(ax0, theta_rem_start, theta_rem_end, "Dreaming", arrow_color='#959FEB', r_arrow=1.05)

    wedge_N1 = wedges[2]
    wedge_N2 = wedges[3]
    theta_ls_start = min(wedge_N1.theta1, wedge_N2.theta1) + gap_deg/2
    theta_ls_end   = max(wedge_N1.theta2, wedge_N2.theta2) - 2*gap_deg/2
    add_group_arc(ax0, theta_ls_start, theta_ls_end, "Light Sleep", arrow_color='#959FEB', r_arrow=1.05, is_ls=True)

    wedge_N3 = wedges[4]
    theta_deep_start = wedge_N3.theta1 + gap_deg/2
    theta_deep_end   = wedge_N3.theta2 - 2*gap_deg/2
    if wedge_N3.theta2 == wedge_N3.theta1: # If N3 is empty
        theta_deep_start = wedge_N3.theta1
        theta_deep_end = wedge_N3.theta2
    add_group_arc(ax0, theta_deep_start, theta_deep_end, "Deep Sleep", arrow_color='#959FEB', r_arrow=1.05, is_deep=True)   

    window = 30
    smooth_hypno = np.empty_like(hypno_mapped, dtype=float)
    half_window = window // 2
    for i in range(len(hypno_mapped)):
        start = max(0, i - half_window)
        end = min(len(hypno_mapped), i + half_window + 1)
        smooth_hypno[i] = np.mean(hypno_mapped[start:end])

    for i in range(len(epochs)-1):
        stage_val = int(np.rint(smooth_hypno[i]))
        color = hypno_colors.get(stage_val, "blue")
        ax1.plot(epochs[i:i+2], smooth_hypno[i:i+2], color=color, linewidth=2)

    for row in range(1, 5):
        ax1.hlines(row, 0, len(epochs), color="white", linewidth=1, alpha=0.1)

    ax1.spines["left"].set_visible(True)
    ax1.spines["left"].set_color("white")
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["bottom"].set_color("white")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    ax1.set_yticks([4, 3, 2, 1, 0])
    ax1.set_yticklabels(["Awake", "REM", "N1", "N2", "N3"], color='white')
    ax1.set_ylim(-0.5, 4.5)

    xticks = np.arange(0, len(epochs)+1, 60)
    xtick_labels = [f"{int((t*0.5)//60)}h" if (t % 120 == 0) else "" for t in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels, color='white')
    ax1.set_xlim(0, len(epochs))

    ax1.grid(False)

    legend_handles = [
        Patch(facecolor=hypno_colors[4], label="Awake"),
        Patch(facecolor=hypno_colors[3], label="REM"),
        Patch(facecolor=hypno_colors[2], label="N1"),
        Patch(facecolor=hypno_colors[1], label="N2"),
        Patch(facecolor=hypno_colors[0], label="N3")
    ]
    ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=5, frameon=False)

    for text in ax1.get_legend().get_texts():
        text.set_color("white")

    plt.tight_layout()
    return fig, total_sleep_str

class SleepStageReportPage(QWidget):
    def __init__(self, config: SessionConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.file_reader = FileReader(self.config)
        self.parent_app = parent  

        self.setStyleSheet("background-color: #1A0033;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.greeting_label = QLabel("Good Morning Brandon! Here is your sleep report!")
        self.greeting_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.greeting_label.setStyleSheet("color: white;")
        self.layout.addWidget(self.greeting_label)

        self.figure_label = QLabel()
        self.figure_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.figure_label)

        self.asleep_label = QLabel("")
        self.asleep_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.asleep_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.asleep_label)

        self.back_button = QPushButton("Back")
        self.back_button.setFont(QFont("Arial", 20))
        self.back_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        self.back_button.clicked.connect(self.go_back_home)
        self.layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

    def generate_and_display_report(self, file_reader: FileReader):
        fig, total_sleep_str = generate_sleep_figure(file_reader)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        buf.close()
        plt.close(fig)

        self.figure_label.setPixmap(pixmap)
        self.asleep_label.setText(f"You were asleep for {total_sleep_str}")
        self.asleep_label.setStyleSheet("color: white;")

    def go_back_home(self):
        if self.parent_app:
            self.parent_app.stacked_widget.setCurrentWidget(self.parent_app.main_page)
            self.parent_app.showFullScreen()

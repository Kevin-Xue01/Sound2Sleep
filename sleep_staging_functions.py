# Your MNE imports for the staging
import mne
import yasa
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle
from mne.datasets.sleep_physionet.age import fetch_data
import matplotlib.pyplot as plt
import os
import io
from matplotlib.patches import Patch

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QStackedWidget, QSizePolicy, QTextEdit, QScrollArea
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

def generate_sleep_figure():
    data_dir = "./sleep_data"
    os.makedirs(data_dir, exist_ok=True)
    records = fetch_data(subjects=[0], recording=[1], path=data_dir)
    edf_file = records[0]  # First file

    raw = mne.io.read_raw_edf(edf_file[0], preload=True)
    raw.resample(100)  # Downsample to 100 Hz

    eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch or "CH" in ch]
    selected_channel = "CH 2" if "CH 2" in eeg_channels else eeg_channels[0]
    raw.pick_channels([selected_channel])

    sls = yasa.SleepStaging(raw, eeg_name=selected_channel)
    hypno_pred = sls.predict()
    hypno_pred = yasa.hypno_str_to_int(hypno_pred)
    hypno_pred = hypno_pred[900:1750]

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
        # Each epoch is 30 seconds, so total sleep time (TST) in minutes:
        TST = sleep_count * 0.5  
        # Define a recommended minimum sleep duration (e.g., 7 hours = 420 minutes)
        sleep_duration_factor = min(1, TST / 420)
        
        # Calculate proportions for each sleep stage
        rem_pct = counts["REM"] / sleep_count
        n1_pct  = counts["N1"]  / sleep_count
        n2_pct  = counts["N2"]  / sleep_count
        n3_pct  = counts["N3"]  / sleep_count
        
        # Sleep quality factor weighted more toward restorative sleep:
        # For instance: deep sleep (N3) 35%, REM 25%, light sleep (N1+N2) 40%
        sleep_quality_factor = 0.35 * n3_pct + 0.25 * rem_pct + 0.40 * (n1_pct + n2_pct)
        
        # Composite sleep score: a blend of sleep duration and sleep quality
        # This yields a score on a 0-100 scale.
        sleep_score = int((0.5 * sleep_duration_factor + 0.5 * sleep_quality_factor) * 100)
    else:
        sleep_score = 0

    stage_map = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    hypno_mapped = np.array([stage_map[s] for s in hypno_pred])
    hypno_colors = {
        4: "#FFC179",  # Awake
        3: "#F195AC",  # REM
        2: "#E3F195",  # N1
        1: "#95F1A8",  # N2
        0: "#95C8F1",  # N3
    }
    epochs = np.arange(len(hypno_mapped))

    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))

    fig.patch.set_facecolor("white")
    ax0.set_facecolor("white")
    ax1.set_facecolor("white")

    width = 0.3
    wedges, _ = ax0.pie(
        data, colors=donut_colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=width, edgecolor='white')
    )
    ax0.axis('equal')  # Keep circle shape
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set_frame_on(False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax0.text(0.5, 1.1, "Welcome to your sleep report!", ha='center', va='bottom',
             fontsize=14, fontweight='bold', color='black', transform=ax0.transAxes)
    ax0.text(0.5, 1.02, f"Sleep Score: {sleep_score}", ha='center', va='bottom',
             fontsize=13, fontweight='bold', color='black', transform=ax0.transAxes)
    
    for wedge, label_name in zip(wedges, stage_names):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        rad = np.deg2rad(angle)
        r = 1 - (width / 2)
        x = r * np.cos(rad)
        y = r * np.sin(rad)
        ax0.text(x, y, label_name, ha='center', va='center',
                 fontsize=10, color='black')

    stage_lines = [f"{stg}: {format_time(counts[stg])}" for stg in stage_names]
    stage_text = "\n".join(stage_lines)
    ax0.text(0.5, -0.15, stage_text, ha='center', va='top', fontsize=11,
             color='black', transform=ax0.transAxes)
        # Insert total sleep time text in the center of the donut chart
    ax0.text(0, 0, f"Total Sleep Time:\n{total_sleep_str}", 
             ha='center', va='center', fontsize=12, color='black', fontweight='bold')


    def add_group_arc(theta_start, theta_end, group_label, arrow_color, r_arrow=1.05, is_ls=False, is_deep=False):
        arc = Arc((0, 0), width=2*r_arrow, height=2*r_arrow,
                  angle=0, theta1=theta_start, theta2=theta_end, lw=2, color=arrow_color)
        ax0.add_patch(arc)
        
        mid_angle = (theta_start + theta_end) / 2.0
        mid_rad = np.deg2rad(mid_angle)
        x_text = (r_arrow + 0.3) * np.cos(mid_rad)
        y_text = (r_arrow + 0.3) * np.sin(mid_rad)
        if is_ls:
            x_text = (r_arrow + 0.15) * np.cos(mid_rad)
            y_text = (r_arrow + 0.15) * np.sin(mid_rad)
        elif is_deep:
            x_text = (r_arrow + 0.35) * np.cos(mid_rad)
            y_text = (r_arrow + 0.35) * np.sin(mid_rad)
        ax0.text(x_text, y_text, group_label, ha='center', va='center',
                 fontsize=12, color='black', fontweight='bold')
        
        end_rad = np.deg2rad(theta_end)
        x_end = r_arrow * np.cos(end_rad)
        y_end = r_arrow * np.sin(end_rad)
        dx = 0.1 * (-np.sin(end_rad))
        dy = 0.1 * (np.cos(end_rad))
        arrow = FancyArrowPatch((x_end, y_end), (x_end + dx, y_end + dy),
                                arrowstyle='-|>', mutation_scale=12,
                                color=arrow_color, lw=2, shrinkA=0, shrinkB=0)
        ax0.add_patch(arrow)

    gap_deg = 5
    wedge_REM = wedges[1]
    theta_rem_start = wedge_REM.theta1 + gap_deg/2
    theta_rem_end   = wedge_REM.theta2 - 2*gap_deg/2
    add_group_arc(theta_rem_start, theta_rem_end, "Dreaming", arrow_color='#959FEB', r_arrow=1.05)

    wedge_N1 = wedges[2]
    wedge_N2 = wedges[3]
    theta_ls_start = min(wedge_N1.theta1, wedge_N2.theta1) + gap_deg/2
    theta_ls_end   = max(wedge_N1.theta2, wedge_N2.theta2) - 2*gap_deg/2
    add_group_arc(theta_ls_start, theta_ls_end, "Light Sleep", arrow_color='#959FEB', r_arrow=1.05, is_ls=True)

    wedge_N3 = wedges[4]
    theta_deep_start = wedge_N3.theta1 + gap_deg/2
    theta_deep_end   = wedge_N3.theta2 - 2*gap_deg/2
    add_group_arc(theta_deep_start, theta_deep_end, "Deep Sleep", arrow_color='#959FEB', r_arrow=1.05, is_deep=True)

    for spine in ax1.spines.values():
        spine.set_visible(False)

    for i, stage in enumerate(hypno_mapped):
        top_y = stage + 1
        rect = Rectangle((i, 0), 1, top_y, color=hypno_colors[stage], edgecolor="white")
        ax1.add_patch(rect)

    for row in range(1, 5):
        ax1.hlines(row, 0, len(epochs), color="black", linewidth=1)

     # Set visible left and bottom spines
    ax1.spines["left"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    ax1.set_yticks([5, 4, 3, 2, 1])
    ax1.set_yticklabels(["Awake", "REM", "N1", "N2", "N3"])

    # Set x ticks: every 60 epochs (30 minutes) but only label if it's a whole hour (i.e. tick%120==0)
    xticks = np.arange(0, len(epochs)+1, 60)
    xtick_labels = [f"{int((t*0.5)//60)}h" if (t % 120 == 0) else "" for t in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    ax1.set_xlim(0, len(epochs))
    ax1.set_ylim(0, 5)
    ax1.set_title("", fontsize=14, pad=10)
    ax1.grid(False)

    legend_handles = [Patch(facecolor=hypno_colors[4], edgecolor='white', label="Awake"),
                      Patch(facecolor=hypno_colors[3], edgecolor='white', label="REM"),
                      Patch(facecolor=hypno_colors[2], edgecolor='white', label="N1"),
                      Patch(facecolor=hypno_colors[1], edgecolor='white', label="N2"),
                      Patch(facecolor=hypno_colors[0], edgecolor='white', label="N3")]
    ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=5, frameon=False)

    plt.tight_layout()
    return fig, total_sleep_str

class SleepStageReportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent  
        self.setStyleSheet("background-color: white;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.greeting_label = QLabel("Good Morning Brandon! Here is your sleep report!")
        self.greeting_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.greeting_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.greeting_label)

        self.figure_label = QLabel()
        self.figure_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.figure_label)

        self.asleep_label = QLabel("")
        self.asleep_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.asleep_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.asleep_label)

        self.back_button = QPushButton("Back")
        self.back_button.setFont(QFont("Arial", 16))
        self.back_button.setStyleSheet("background-color: #3A1D92; color: white; padding: 10px; border-radius: 10px;")
        self.back_button.clicked.connect(self.go_back_home)
        self.layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

    def generate_and_display_report(self):
        fig, total_sleep_str = generate_sleep_figure()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        buf.close()
        plt.close(fig)

        self.figure_label.setPixmap(pixmap)
        self.asleep_label.setText(f"You were asleep for {total_sleep_str}")

    def go_back_home(self):
        if self.parent_app:
            self.parent_app.showNormal()
            self.parent_app.stacked_widget.setCurrentWidget(self.parent_app.main_page)

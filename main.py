'''
GUI for running the Closed Loop Auditory Stimulation experiment
Author: Simeon Wong
'''

import sys
import traceback

# Qt Framework
from typing import Callable
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRunnable, QTimer, Qt, QThreadPool, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QStaticText
from PyQt5.QtWidgets import QComboBox, QDateTimeEdit, QWidget, QFileDialog
import qdarkstyle

import os
import io
import json
import struct
import time
from enum import Enum
from typing import Optional
import logging
import psutil
import yaml

# Plot stuff
import matplotlib, matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# Math stuff
import numpy as np

# CLAS algorithm
import QCLASAlgo

# Misc
import datetime

# for Pushover
import requests
import yaml


class Main(QtWidgets.QMainWindow):
    def __init__(self, parent: QWidget | None = ..., flags: Qt.WindowFlags | Qt.WindowType = ...) -> None:
        super().__init__(parent, flags)

if __name__ == "__main__":
    App = QtWidgets.QApplication(sys.argv)
    window = Main()
    sys.exit(App.exec())
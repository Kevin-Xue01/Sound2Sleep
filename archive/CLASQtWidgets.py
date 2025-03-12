from typing import Optional
from PyQt5 import QtGui
import PyQt5.QtWidgets
from PyQt5.QtCore import QTimer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from enum import Enum


class QStatusLight(PyQt5.QtWidgets.QLabel):

    class State(Enum):
        BLANK = 0
        OK = 1
        WORKING = 2
        STOPPED = 3

    state: State

    def __init__(self, size_px: int = 42, *args, **kwargs):
        super().__init__(' ', *args, **kwargs)
        self.setGeometry(0, 0, size_px, size_px)
        self.set_state(QStatusLight.State.BLANK)

        # redraw
        self.update()

    def set_state(self, state: State):
        self.state = state

        if state == QStatusLight.State.BLANK:
            self.setStyleSheet(
                f'background-color: none; border-width: 4px; border-color: white; border-style: solid; border-radius: {self.size().width()/2:f}px;'
            )
        elif state == QStatusLight.State.OK:
            self.setStyleSheet(
                f'background-color: #076E00; border-width: 4px; border-color: #055C00; border-style: solid; border-radius: {self.size().width()/2:f}px;'
            )
        elif state == QStatusLight.State.WORKING:
            self.setStyleSheet(
                f'background-color: none; border-width: 4px; border-color: #E59600; border-style: solid; border-radius: {self.size().width()/2:f}px;'
            )
        elif state == QStatusLight.State.STOPPED:
            self.setStyleSheet(
                f'background-color: none; border-width: 4px; border-color: #4F416D; border-style: solid; border-radius: {self.size().width()/2:f}px;'
            )


class QResetButton(PyQt5.QtWidgets.QPushButton):
    timer: Optional[QTimer] = None
    progress: int = 0
    target_progress: int = 0
    timer_interval: int = 100

    reset = PyQt5.QtCore.pyqtSignal()

    def __init__(self, hold_time_s: float = 3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_progress = hold_time_s * 1000 / self.timer_interval

        self.updateStyle()
        self.pressed.connect(self.mousePressEvent)

    def updateStyle(self):
        pct = self.progress / self.target_progress
        if (pct == 0):
            bg = 'background-color: #8F3A00;'
        else:
            # pct = 1-pct
            bg = f'background: qlineargradient(spread:pad, x1:0 y1:{1-pct:f}, x2:0 y2:{1-pct+0.0001:f}, stop:0 white, stop:1 #8F3A00);'

        self.setStyleSheet(f'{bg} color: white; font-size: 30px;')

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        super().mousePressEvent(e)
        self.timer = QTimer()
        self.timer.timeout.connect(self.timerFired)
        self.timer.start(self.timer_interval)
        self.progress = 0

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(e)

        if self.timer is not None:
            self.timer.stop()
            self.progress = 0
            self.updateStyle()

    def timerFired(self):
        self.progress += 1

        if self.progress >= self.target_progress:
            self.progress = 0
            self.timer.stop()
            self.reset.emit()

        self.updateStyle()


class QCPlot(FigureCanvasQTAgg):

    def __init__(self, parent=None, dpi=200):
        # initialize the figure
        wsize = parent.size()
        self.fig = matplotlib.figure.Figure(figsize=(wsize.width() / dpi,
                                                     wsize.height() / dpi),
                                            dpi=dpi)  #type:ignore
        self.axes = self.fig.add_subplot(111)
        self.axes.set_position((0.05, 0.18, 0.92, 0.77))
        # plt.axis('off')
        self.axes.set_axis_off()

        s = super(QCPlot, self)
        s.__init__(self.fig)
        self.setParent(parent)

        s.setSizePolicy(PyQt5.QtWidgets.QSizePolicy.Expanding,
                        PyQt5.QtWidgets.QSizePolicy.Expanding)
        s.updateGeometry()

        self.figure.canvas.setStyleSheet("background-color:transparent;")
        self.figure.patch.set_facecolor('None')

    def clear(self):
        self.axes.cla()

    def plot(self, time, data):
        self.plt = self.axes.plot(time, data.T, lw=2, c='white', alpha=0.8)
        self.pltd = self.axes.scatter(time[-1], data[-1], s=100, c='white')
        # self.axes.set_ylim(-500, 500)
        self.draw()

    def update_data(self, data):
        # for kk, ln in enumerate(self.plt):
        #     ln.set_ydata(data[kk, :])
        self.plt[0].set_ydata(data)
        self.draw()

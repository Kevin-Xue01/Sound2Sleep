"""
Starting a Stream

This example shows how to search for available Muses and
create a new stream
"""
from threading import Thread
from time import sleep

import matplotlib
import numpy as np
import seaborn as sns
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import firwin, lfilter, lfilter_zi

from muselsl import list_muses, stream
from muselsl.constants import (
    LSL_EEG_CHUNK,
    LSL_SCAN_TIMEOUT,
    VIEW_BUFFER,
    VIEW_SUBSAMPLE,
)
from muselsl.viewer_v1 import LSLViewer


def start_stream(address):
    """Function to start the Muse stream in a separate thread."""
    stream(address)

def view(window, scale, refresh, figure, backend, version=1):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.int16(figure.split('x'))

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
    lslv = LSLViewer(streams[0], fig, axes, window, scale)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toogle full screen : f
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    lslv.start()
    matplotlib.pyplot.show()
    
if __name__ == "__main__":
    muses = list_muses()

    if not muses:
        print('No Muses found')
    else:
        # Start streaming in a separate thread
        stream_thread = Thread(target=start_stream, args=(muses[0]['address'],), daemon=True)
        stream_thread.start()

        # Start the viewer in the main thread
        view(5, 100, 0.2, "15x6", 'TkAgg')
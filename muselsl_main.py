import asyncio
import threading

import matplotlib
import numpy as np
import seaborn as sns
from pylsl import resolve_byprop

from muselsl import list_muses, stream
from muselsl.constants import LSL_SCAN_TIMEOUT
from muselsl.viewer_v1 import LSLViewer


def start_stream(address):
    """Run the Muse LSL stream inside an asyncio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream(address))

def view(window, scale, refresh, figure, backend, version=1):
    """Start the EEG data viewer in the main thread."""
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.array(figure.split('x'), dtype=int)

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")
    
    print("Start acquiring data.")
    
    fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
    lslv = LSLViewer(streams[0], fig, axes, window, scale)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                toggle full screen : f
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
        # Start streaming in a separate thread with its own event loop
        stream_thread = threading.Thread(target=start_stream, args=(muses[0]['address'],), daemon=True)
        stream_thread.start()

        # Start the viewer in the main thread
        view(window=5, scale=100, refresh=0.2, figure="10x5", backend="TkAgg")
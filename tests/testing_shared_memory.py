import multiprocessing as mp
import time
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Constants
SAMPLE_RATE = 256  # Hz
WINDOW_SIZE = 10  # seconds
TOTAL_SAMPLES = SAMPLE_RATE * WINDOW_SIZE
NUM_CHANNELS = 4  # Assuming 4 EEG channels
CHUNK_SIZE = 12  # Incoming data chunk size
SHM_NAME = "eeg_shared_memory"

# Producer: Simulates real-time EEG data acquisition
def producer():
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=TOTAL_SAMPLES * NUM_CHANNELS * 4)  # float32 = 4 bytes
    data_array = np.ndarray((TOTAL_SAMPLES, NUM_CHANNELS), dtype=np.float32, buffer=shm.buf)
    
    try:
        while True:
            new_data = np.random.randn(CHUNK_SIZE, NUM_CHANNELS).astype(np.float32)  # Simulated EEG data
            data_array[:-CHUNK_SIZE] = data_array[CHUNK_SIZE:]
            data_array[-CHUNK_SIZE:] = new_data
            time.sleep(CHUNK_SIZE / SAMPLE_RATE)  # Simulate real-time chunk processing
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        shm.unlink()

# Consumer: Reads from shared memory and plots EEG data in real-time
def consumer():
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    data_array = np.ndarray((TOTAL_SAMPLES, NUM_CHANNELS), dtype=np.float32, buffer=shm.buf)
    
    import matplotlib
    matplotlib.use('QtAgg')
    sns.set_theme(style="whitegrid")
    sns.despine(left=True)
    
    fig, ax = plt.subplots(figsize=[15, 6])
    lines = []
    impedances = np.zeros(NUM_CHANNELS)
    time_axis = np.linspace(-WINDOW_SIZE, 0, TOTAL_SAMPLES // 2)  # X-axis from -10s to 0s
    
    for i in range(NUM_CHANNELS):
        line, = ax.plot(time_axis, np.zeros(TOTAL_SAMPLES // 2) - i, lw=1)
        lines.append(line)
    
    ax.set_ylim(-NUM_CHANNELS + 0.5, 0.5)
    ax.set_xlabel('Time (s)')
    ax.xaxis.grid(False)
    ax.set_yticks(np.arange(0, -NUM_CHANNELS, -1))
    ax.set_yticklabels([f'Channel {i+1} - {impedance:2f}' for i, impedance in enumerate(impedances)])
    
    plt.show(block=False)  # Prevents locking focus
    
    try:
        while plt.fignum_exists(fig.number):  # Keeps updating only if the figure exists
            for i, line in enumerate(lines):
                line.set_ydata(data_array[::2, i] - i)  # Offset each channel
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    p = mp.Process(target=producer)
    p.start()
    
    time.sleep(2)
    try:
        consumer()
    finally:
        p.terminate()
        p.join()

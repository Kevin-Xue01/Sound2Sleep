import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import threading
import argparse
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import random

SRATE = 256 # [Hz]

class LSLStreamSimulator:
    def __init__(self, name, stream_type, num_channels, base_chunk_size, ics: bool):
        self.name = name
        self.stream_type = stream_type
        self.num_channels = num_channels
        self.base_chunk_size = base_chunk_size
        self.ics = ics

        self.info = StreamInfo(name, stream_type, num_channels)

        self.outlet = StreamOutlet(self.info)

    def simulate_data(self, chunk_size):
        data_chunk = np.random.rand(chunk_size, self.num_channels).astype(np.float32)
        return data_chunk
    
    def determine_chunk_size(self):
        probabilities = [0.85, 0.1, 0.05]  
        size_multiplier = random.choices([1, 2, 3], probabilities)[0] 
        return self.base_chunk_size * size_multiplier

    def push_data(self):
        while True:
            chunk_size = self.determine_chunk_size(self.base_chunk_size) if self.ics else self.base_chunk_size

            data_chunk = np.array(self.simulate_data(chunk_size))
            self.outlet.push_chunk(data_chunk.tolist())

            sleep_time = chunk_size / SRATE  
            time.sleep(sleep_time)


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Muse S LSL Simulator")
    
    # Add the '--irregular_chunk_size' flag
    parser.add_argument('--irregular_chunk_size', '--ICS', action='store_true', 
                        help="Enable irregular chunk sizes")
    
    args = parser.parse_args()

    # Parameters for the EEG stream
    eeg_simulator = LSLStreamSimulator(name="EEG", stream_type="EEG", num_channels=4, base_chunk_size=12, ics=args.irregular_chunk_size)
    # Parameters for the Accelerometer stream
    accel_simulator = LSLStreamSimulator(name="Accelerometer", stream_type="ACC", num_channels=3, base_chunk_size=6, ics=args.irregular_chunk_size)
    # Parameters for the PPG stream
    ppg_simulator = LSLStreamSimulator(name="PPG", stream_type="PPG", num_channels=3, base_chunk_size=6, ics=args.irregular_chunk_size)

    # Simulate streams in separate threads
    eeg_thread = threading.Thread(target=eeg_simulator.push_data, daemon=True)
    accel_thread = threading.Thread(target=accel_simulator.push_data, daemon=True)
    ppg_thread = threading.Thread(target=ppg_simulator.push_data, daemon=True)

    # Start the threads
    eeg_thread.start()
    accel_thread.start()
    ppg_thread.start()

    # Keep the main thread alive while the streams are running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the simulator...")

if __name__ == '__main__':
    main()
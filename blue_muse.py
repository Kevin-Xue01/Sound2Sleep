import subprocess
import traceback
import time
import numpy as np
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSignal, QObject, QTimer
from pylsl import StreamInlet, resolve_stream, resolve_byprop, StreamInfo
from datetime import datetime
from utils import find_procs_by_name, StreamType, BlueMuseSignal


class BlueMuse(QObject):
    def __init__(self, data_signal: BlueMuseSignal, sleep_duration: float=12/256):
        super().__init__()
        self.data_signal = data_signal
        self.sleep_duration = sleep_duration

        self.stream_infos: dict[StreamType, StreamInfo] = {}
        self.stream_inlets: dict[StreamType, StreamInlet] = {}
        # self.files = {}
        self.no_data_count = 0
        self.reset_attempt_count = 0

        # start bluemuse if not already started
        subprocess.call('start bluemuse:', shell=True)
        subprocess.call('start bluemuse://setting?key=primary_timestamp_format!value=BLUEMUSE', shell=True)
        subprocess.call('start bluemuse://setting?key=channel_data_type!value=float32', shell=True)
        subprocess.call('start bluemuse://setting?key=eeg_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=accelerometer_enabled!value=true', shell=True)
        subprocess.call('start bluemuse://setting?key=gyroscope_enabled!value=false', shell=True)
        subprocess.call('start bluemuse://setting?key=ppg_enabled!value=true', shell=True)

    def lsl_reload(self):
        ''' 
        Resolve all 3 LSL streams from the Muse S.
        This function blocks for up to 10 seconds.
        '''
        print('\n\n=== ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ===\n')
        allok = True
        self.stream_infos = {}
        for streamtype in StreamType:
            result = resolve_byprop('type', streamtype.value, timeout=10)

            if result:
                self.stream_infos[streamtype] = result[0]
                print('%s OK.' % streamtype.value)
            else:
                print('%s not found.' % streamtype.value)
                allok = False
        return allok
    

    def lsl_timer_callback(self):
        ''' 
        Get data from LSL streams and route it to the right place (plot, files, phase tracker).
        Callback for lsl_timer.
        '''
        for streamtype, streaminlet in self.stream_inlets.items():
            try:
                data, times = streaminlet.pull_chunk()
                data = np.array(data)

                if len(times) > 0:
                    self.no_data_count = 0

                    start_time = times[0]

                    # Calculate the time step between consecutive samples
                    time_step = 1.0 / 256

                    # Generate unique timestamps for each sample within the chunk
                    unique_times = np.array([start_time + i * time_step for i in range(len(data))])
                    # print(unique_times, np.array(data))
                    # Emit the data with the generated unique timestamps
                    self.data_signal.update_data.emit(streamtype, unique_times, np.array(data))

                    # # store the data
                    # self.plots[d].add_data(chunk)

                    # # submit EEG data to the PLL
                    # if d == 'EEG':
                    #     # _, ts_ref, ts_lockbin = self.pll.process_block(chunk[:, 0])
                    #     self.clas_algo.new_data(chunk[:, 0])

                    # self.files[d].write('NCHK'.encode('ascii'))
                    # self.files[d].write(chunk.dtype.char.encode('ascii'))
                    # self.files[d].write(np.array(chunk.shape).astype(np.uint32).tobytes())
                    # self.files[d].write(np.array(times).astype(np.double).tobytes())
                    # self.files[d].write('TTTT'.encode('ascii'))
                    # self.files[d].write(chunk.tobytes(order='C'))
                    # print(chunk)

                else:
                    self.no_data_count += 1

                    # if no data after 2 seconds, attempt to reset and recover
                    if self.no_data_count > 20:
                        self.lsl_reset_stream_step1()
                time.sleep(self.sleep_duration)
                
            except Exception as ex:
                # construct traceback
                tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                tbstring.insert(0, '=== ' + str(datetime.now()) + ' ===')

                # print to screen and error log file
                print('\n'.join(tbstring))

    def lsl_reset_stream_step1(self):
        self.no_data_count = 0  # reset no data counter
        self.stop_streaming()  # stop data pulling loop

        # restart bluemuse streaming, wait, and restart
        subprocess.call('start bluemuse://stop?stopall', shell=True)
        QTimer.singleShot(3000, self.lsl_reset_stream_step2)

    def lsl_reset_stream_step2(self):
        ''' 
        Try to restart streams the lsl pull timer.
        Part of interrupted stream restart process.
        '''
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        QTimer.singleShot(3000, self.lsl_reset_stream_step3)

    def lsl_reset_stream_step3(self):
        reset_success = self.lsl_reload()

        if not reset_success:
            # if we can't get the streams up, try again
            self.reset_attempt_count += 1
            if self.reset_attempt_count < 3:
                self.lsl_reset_stream_step1()
            else:
                self.reset_attempt_count = 0

                # if the stream really isn't working.. kill bluemuse
                for p in find_procs_by_name('BlueMuse.exe'):
                    p.kill()

                # try the reset process again
                QTimer.singleShot(3000, self.lsl_reset_stream_step1)
        else:
            # if all streams have resolved, start polling data again!
            self.reset_attempt_count = 0
            self.start_streaming()

    def run(self):
        self.start_streaming()

    def stop(self):
        self.stop_streaming()

    def start_streaming(self):
        ''' 
        Callback for "Start" button
        Start bluemuse, streams, initialize recording files
        '''

        # initialize bluemuse and try to resolve LSL streams
        subprocess.call('start bluemuse://start?streamfirst=true', shell=True)
        self.lsl_reload()
        #     # self.status.setStyleSheet("background-color: yellow")
        #     # self.status.setText('Unable to connect to Muse S...')
        #     return

        # # initialize metadata file
        # fileroot = uuid.uuid4().hex
        # starttime = datetime.now()
        # self.meta = {
        #     "start_time": starttime.isoformat(),
        #     "data": {},
        #     "fs": {},
        #     "nchan": {},
        #     "error_log": fileroot + '_err.txt'
        # }

        # start the selected steram
        for streamtype, streaminfo in self.stream_infos.items():
            self.stream_inlets[streamtype] = StreamInlet(streaminfo)
            # self.plots[k].init_data(fsample=self.lsl[k].nominal_srate(),
            #                         history_time=8,
            #                         nchan=self.lsl[k].channel_count())

            # # include details in metadata
            # self.meta['data'][k] = fileroot + '_' + k + '.dat'
            # self.meta['fs'][k] = self.lsl[k].nominal_srate()
            # self.meta['nchan'][k] = self.lsl[k].channel_count()
            # curr_path = os.path.join('output', self.meta['data'][k])
            # # os.makedirs(curr_path, exist_ok=True)
            # self.files[k] = open(curr_path, 'wb')

        # # save the metafile
        # with open(os.path.join('output', 'cah_%s.json' % starttime.strftime('%Y%m%dT%H%M%S')), 'w') as f:
        #     json.dump(self.meta, f)

        # # initialize the error log
        # self.files['err'] = open(os.path.join('output', self.meta['error_log']), 'w')
        self.lsl_timer = QTimer()
        self.lsl_timer.timeout.connect(self.lsl_timer_callback)
        self.lsl_timer.start(200)

    def stop_streaming(self):
        ''' 
        Callback for "Stop" button
        Stop lsl chunk timers, GUI update timers, stop streams
        '''
        self.lsl_timer.stop()

        for _, streaminlet in self.stream_inlets.items():
            try:
                streaminlet.close_stream()
            except Exception as ex:
                # construct traceback
                tbstring = traceback.format_exception(type(ex), ex, ex.__traceback__)
                tbstring.insert(0, '=== ' + datetime.now() + ' ===')

                # print to screen and error log file
                print('\n'.join(tbstring))

        subprocess.call('start bluemuse://stop?stopall', shell=True)
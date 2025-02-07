import json
import numpy as np
import os.path


def load(path: str):
    with open(path, 'r') as f:
        meta = json.load(f)

    meta_dir, _ = os.path.split(path)

    data = {'fs': meta['fs'], 'nchan': meta['nchan'], 'data': {}, 'times': {}}
    for k in meta['data'].keys():
        data['data'][k], data['times'][k] = read_data_file(os.path.join(meta_dir, meta['data'][k]))

    return data


def read_data_file(path: str):
    data = []
    times = []
    with open(path, 'rb') as f:
        # try reading to see if we're at the end of the file
        temp = f.read(4)
        while len(temp) > 0:
            # check for the start sequence
            if temp != b'NCHK':
                raise('Start sequence not found at %s' % f.tell())

            # read the data type
            dtype = np.dtype(f.read(1).decode('ascii'))

            # read the data shape
            dshape = np.frombuffer(f.read(8), dtype=np.uint32)

            # read the times
            ctime = np.frombuffer(f.read(8 * dshape[0]), dtype=np.double)
            times.append(ctime)

            # read the separator
            temp = f.read(4)
            if temp != b'TTTT':
                raise('Separator sequence not found at %s' % f.tell())

            # read the data
            cdata = np.frombuffer(f.read(dtype.itemsize * dshape[0] * dshape[1]), dtype=dtype).reshape(dshape, order='C')
            data.append(cdata)

            # read again
            temp = f.read(4)

    data = np.concatenate(data, axis=0)
    times = np.concatenate(times)
    
    return data, times

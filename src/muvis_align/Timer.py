# https://www.geeksforgeeks.org/time-process_time-function-in-python/

import logging
import time


class Timer(object):
    def __init__(self, title, auto_unit=True, verbose=True):
        self.title = title
        self.auto_unit = auto_unit
        self.verbose = verbose
        self.reset()

    def __enter__(self):
        self.ptime_start = time.process_time()
        self.time_start = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            ptime_end = time.process_time()
            time_end = time.time()
            pelapsed = ptime_end - self.ptime_start
            elapsed = time_end - self.time_start
            unit = 'seconds'
            if self.auto_unit and elapsed >= 60:
                pelapsed /= 60
                elapsed /= 60
                unit = 'minutes'
                if elapsed >= 60:
                    pelapsed /= 60
                    elapsed /= 60
                    unit = 'hours'
            logging.info(f'Time {self.title}: {elapsed:.1f} ({pelapsed:.1f}) {unit}')

    def reset(self):
        self.total_ptime = 0
        self.total_time = 0
        self.ntotal = 0

    def start(self):
        self.ptime_start = time.process_time()
        self.time_start = time.time()

    def record(self):
        ptime_end = time.process_time()
        time_end = time.time()
        pelapsed = ptime_end - self.ptime_start
        elapsed = time_end - self.time_start
        self.total_ptime += pelapsed
        self.total_time += elapsed
        self.ntotal += 1

    def get_total_time(self):
        if self.verbose:
            logging.info(f'Time {self.title}: {self.total_time:.1f} ({self.total_ptime:.1f}) seconds #{self.ntotal}')
        return self.total_time, self.total_ptime, self.ntotal

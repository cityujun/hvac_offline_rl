import sys
import numpy as np
import math
from datetime import datetime, timedelta


def show_statistics(title, series, file=sys.stdout):
    res = '{:25} ave={:5,.2f}, min={:5,.2f}, max={:5,.2f}, std={:5,.2f}'.format(title, np.average(series), np.min(series), np.max(series), np.std(series))
    print(res, file=file)


def get_statistics(series):
    return np.average(series), np.min(series), np.max(series), np.std(series)


def show_distrib(title, series):
    dist = [0 for i in range(1000)]
    for v in series:
        idx = int(math.floor(v * 10))
        if idx >= 1000:
            idx = 999
        if idx < 0:
            idx = 0
        dist[idx] += 1
    print(title)
    print('    degree 0.0-0.9 0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9')
    print('    -------------------------------------------------------------------------')
    for t in range(170, 280, 10):
        print('    {:4.1f}C {:5.1%}  '.format(t / 10.0, sum(dist[t:(t+10)]) / len(series)), end='')
        for tt in range(t, t + 10):
            print(' {:5.1%}'.format(dist[tt] / len(series)), end='')
        print('')


#############################
## Below for ibm-datacenter
#############################

# Parse date/time format from EnergyPlus and return datetime object with correction for 24:00 case
def _parse_datetime(dstr):
    # ' MM/DD  HH:MM:SS' or 'MM/DD  HH:MM:SS'
    # Dirty hack
    if dstr[0] != ' ':
        dstr = ' ' + dstr
    #year = 2017
    year = 2013 # for CHICAGO_IL_USA TMY2-94846
    month = int(dstr[1:3])
    day = int(dstr[4:6])
    hour = int(dstr[8:10])
    minute = int(dstr[11:13])
    sec = 0
    msec = 0
    if hour == 24:
        hour = 0
        dt = datetime(year, month, day, hour, minute, sec, msec) + timedelta(days=1)
    else:
        dt = datetime(year, month, day, hour, minute, sec, msec)
    return dt

# Convert list of date/time string to list of datetime objects
def convert_datetime24(dates):
    # ' MM/DD  HH:MM:SS'
    dates_new = []
    for d in dates:
        #year = 2017
        #month = int(d[1:3])
        #day = int(d[4:6])
        #hour = int(d[8:10])
        #minute = int(d[11:13])
        #sec = 0
        #msec = 0
        #if hour == 24:
        #    hour = 0
        #    d_new = datetime(year, month, day, hour, minute, sec, msec) + dt.timedelta(days=1)
        #else:
        #    d_new = datetime(year, month, day, hour, minute, sec, msec)
        #dates_new.append(d_new)
        dates_new.append(_parse_datetime(d))
    return dates_new

# Generate x_pos and x_labels
def generate_x_pos_x_labels(dates):
    time_delta  = _parse_datetime(dates[1]) - _parse_datetime(dates[0])
    x_pos = []
    x_labels = []
    for i, d in enumerate(dates):
        dt = _parse_datetime(d) - time_delta
        if dt.hour == 0 and dt.minute == 0:
            x_pos.append(i)
            if dt.day == 1:
                x_labels.append(dt.strftime('%m/%d'))
            else:
                x_labels.append('')
    return x_pos, x_labels

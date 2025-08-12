import json
import numpy as np


frames_per_sec = 15
sec_per_min = 60
min_per_day = 25
frames_per_minute = frames_per_sec*sec_per_min
frames_per_day = frames_per_sec*sec_per_min*min_per_day


def hello2():
    pass

def load_mice_data(group="main", foldername="raw_mice_data"):
    frame_within = np.loadtxt('../{}/frame_within.csv'.format(foldername), delimiter=",", skiprows=0)
    frame_within_tail_behind = np.loadtxt('../{}/frame_within_tail_behind.csv'.format(foldername), delimiter=",", skiprows=0)
    frame_within_tail_front = np.loadtxt('../{}/frame_within_tail_not_behind.csv'.format(foldername), delimiter=",", skiprows=0)
    bout_start = json.load(open('../{}/bout_start.json'.format(foldername)))
    bout_end = json.load(open('../{}/bout_end.json'.format(foldername)))
    bout_duration = np.loadtxt('../{}/bout_duration_raw.csv'.format(foldername), delimiter=",", skiprows=0)

    bout_start = {int(k): v for k,v in bout_start.items()}
    bout_end = {int(k): v for k,v in bout_end.items()}

    if group == "main":
        frame_within = frame_within[:26, :]
        frame_within_tail_behind = frame_within_tail_behind[:26, :]
        frame_within_tail_front = frame_within_tail_front[:26, :]
        bout_start = {k:v for k,v in bout_start.items() if k < 26}
        bout_end = {k:v for k,v in bout_end.items() if k < 26}
        bout_duration = bout_duration[:26, :]
    
    elif group == "context":
        frame_within = frame_within[26:, :]
        frame_within_tail_behind = frame_within_tail_behind[26:, :]
        frame_within_tail_front = frame_within_tail_front[26:, :]
        bout_start = {k:v for k,v in bout_start.items() if k >= 26}
        bout_end = {k:v for k,v in bout_end.items() if k >= 26}
        bout_duration = bout_duration[26:, :]

    else:
        return None

    ## Sort Animals by Time at Object
    idx_avg = [(i, np.mean(frame_within[i])) for i in range(frame_within.shape[0])]
    idx_avg_sort = sorted(idx_avg, key=lambda t: t[1])
    idx_order = [t[0] for t in idx_avg_sort]

    return frame_within, frame_within_tail_behind, frame_within_tail_front, bout_start, bout_end, bout_duration, idx_order


def moving_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')


def average_time_data(frame_within, frame_within_tail_behind, frame_within_tail_front, frames_per_minute, sample_rate, avg_multiplier):
    avg_window = int(frames_per_minute*avg_multiplier)
    
    c = 1 if avg_multiplier % 1 == 0 else 0
    
    output_shape = (frame_within.shape[0], int(
        np.floor(frame_within.shape[1]/(sample_rate)
                 -np.floor(avg_multiplier*frames_per_minute/sample_rate))+c))

    near = np.zeros(output_shape)
    near_behind = np.zeros(output_shape)
    near_front = np.zeros(output_shape)

    for i in range(frame_within.shape[0]):
        near[i, :] = moving_average(frame_within[i], avg_window)[0::sample_rate]
        near_behind[i, :] = moving_average(frame_within_tail_behind[i], avg_window)[0::sample_rate]
        near_front[i, :] = moving_average(frame_within_tail_front[i], avg_window)[0::sample_rate]
        
    return near, near_behind, near_front


def average_nonzero_duration(frame_within, bout_duration, frames_per_minute, sample_rate, avg_multiplier, interpolate=False):
        
    avg_window = int(frames_per_minute*avg_multiplier)
    output_shape = (frame_within.shape[0], int(
        np.floor(frame_within.shape[1]/(sample_rate)
                 -np.floor(avg_multiplier*frames_per_minute/sample_rate))+1))
    
    bout_duration_avg = np.zeros(output_shape)
    
    for i in range(output_shape[0]):
        nz_idxs = bout_duration[i].nonzero()[0]

        for j in range(0, output_shape[1]):
            bout_duration_avg[i, j] = np.mean(bout_duration[i][nz_idxs[np.logical_and(nz_idxs>=j*avg_window, nz_idxs<(j+1)*avg_window)]])
        
    output = np.nan_to_num(bout_duration_avg, nan=0.0)

    if interpolate:
        output_interp = np.zeros([output_shape[0], 150])
        for i in range(output_shape[0]):
            output_interp[i] = np.interp(np.arange(0, 150), np.arange(avg_multiplier, 150+1, avg_multiplier), output[i])
            
        return output_interp
    else:
        return output
    

def average_freq(frame_within, bout_start, frames_per_minute, sample_rate, avg_multiplier, dataset="main"):
    avg_window = int(frames_per_minute*avg_multiplier)
    output_shape = (frame_within.shape[0], int(np.floor(150-avg_multiplier)+1))

    bout_freq = np.zeros(output_shape)
    
    for i in range(len(bout_start)):
        bout_one_hot = np.zeros(frame_within[0].shape)

        if dataset == "main":
            bout_one_hot[bout_start[i]] = 1.0
        elif dataset == "context":
            bout_one_hot[bout_start[i+26]] = 1.0
        else:
            return None

        bout_freq[i, :] = moving_average(bout_one_hot, avg_window)[0::sample_rate]

    return bout_freq
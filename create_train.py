import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import math
import os

# RAW_CSV_INPUTS_DIR = './RAW_DATA/sample_CSV_files/'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/large_CSV_files/'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/train_CSV_files/'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/test_CSV_files/'
RAW_CSV_INPUTS_DIR = './RAW_DATA/recent-mission/'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/simulator_CSV_files'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/simulator_curr_CSV_files'
# RAW_CSV_INPUTS_DIR = './RAW_DATA/sim_retrain_CSV_files'

PROCESSED_CSV_FILES = ['Depth.csv', 'Distance.csv', 'EstimatedState.csv', 'Rpm.csv']

COLOR_MAP = {'TRACKING': 'green', 'CLIMBING': 'yellow', 'AVOIDING': 'red', 'UNSTEADY': 'black', 'SURFACED': 'blue'}

k = 0

SAMPLING_RATE = 0.2
TIME_WINDOW_SIZE_SEC = 10
TIME_WINDOW_NUM_SAMPLE = int((1 / SAMPLING_RATE) * TIME_WINDOW_SIZE_SEC)

print('Samples per time-window:', TIME_WINDOW_NUM_SAMPLE)

def plot_input_data(sensor_data):
    fig, ax = plt.subplots()
    ax.plot(sensor_data['timestamp'], sensor_data['DVL-0'], color='blue')
    ax.plot(sensor_data['timestamp'], sensor_data['DVL-1'], color='green')
    ax.plot(sensor_data['timestamp'], sensor_data['DVL-2'], color='red')
    ax.plot(sensor_data['timestamp'], sensor_data['DVL-3'], color='yellow')
    ax.plot(sensor_data['timestamp'], sensor_data['DVL-filtered'], color='cyan')
    ax.plot(sensor_data['timestamp'], sensor_data['Echo'], color='black')
    ax.plot(sensor_data['timestamp'], sensor_data['theta'], color='magenta')
    plt.show()

def plot_labeled_data(sensor_data_automatically_labeled):
    global k
    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    
    #plot the automatically labeled data
    ax[0].scatter(sensor_data_automatically_labeled['timestamp'], sensor_data_automatically_labeled['Echo'], c=[COLOR_MAP[label] for label in sensor_data_automatically_labeled['Label'].tolist()])
    ax[0].set_title('Echo sounder sensor data')
    ax[0].set_ylabel('Echo sounder value (m)')
    ax[0].set_xlabel('Timestamp (s)')

    #plot the automatically labeled data
    ax[1].scatter(sensor_data_automatically_labeled['timestamp'], sensor_data_automatically_labeled['DVL-filtered'], c=[COLOR_MAP[label] for label in sensor_data_automatically_labeled['Label'].tolist()])
    ax[1].set_title('DVL-filtered sensor data')
    ax[1].set_ylabel('DVL-filtered value (m)')
    ax[1].set_xlabel('Timestamp (s)')

    ax[2].scatter(sensor_data_automatically_labeled['timestamp'], sensor_data_automatically_labeled['depth'], c=[COLOR_MAP[label] for label in sensor_data_automatically_labeled['Label'].tolist()])
    ax[2].set_title('Depth sensor data')
    ax[2].set_ylabel('Depth value (m)')
    ax[2].set_xlabel('Timestamp (s)')
    
    handles, labels = plt.gca().get_legend_handles_labels()

    line1 = Line2D([0], [0], linewidth=5, label='TRACKING', color='green')
    line2 = Line2D([0], [0], linewidth=5, label='CLIMBING', color='yellow')
    line3 = Line2D([0], [0], linewidth=5, label='AVOIDING', color='red')
    line4 = Line2D([0], [0], linewidth=5, label='UNSTEADY', color='black')
    line5 = Line2D([0], [0], linewidth=5, label='SURFACED', color='blue')

    handles.extend([line1, line2, line3, line4, line5])

    fig.legend(handles=handles, loc='upper right')
    # fig.tight_layout()
    # fig.savefig('labelled_data_' + str(k) + '.png')
    plt.show()

def get_closest_rpm(timestamp, rpm_value_key, rpm_df):
    closest = rpm_df.iloc[(rpm_df['timestamp']-timestamp).abs().argmin()]
    return closest[rpm_value_key]

def get_closest_depth(timestamp, depth_val_key, depth_entity_key, depth_df):
    closest = depth_df.iloc[(depth_df['timestamp']-timestamp).abs().argmin()]
    return closest[depth_val_key]
    
    # closest_timestamp_df_ori = depth_df.iloc[(depth_df['timestamp']-timestamp).abs().argsort()]
    # closest_timestamp_df = closest_timestamp_df_ori.loc[closest_timestamp_df_ori[entity_key].str.contains('Depth', case=False)]
    # if closest_timestamp_df.empty:
    #     closest_timestamp_df = closest_timestamp_df_ori.loc[closest_timestamp_df_ori[entity_key].str.contains('SmartX', case=False)]
    # if closest_timestamp_df.empty:
    #     closest_timestamp_df = closest_timestamp_df_ori.loc[closest_timestamp_df_ori[entity_key].str.contains('DVL', case=False)]
    
    # return closest_timestamp_df[depth_val_key].tolist()[0]

def get_closest_theta(timestamp, theta_key, est_state_df):
    closest = est_state_df.iloc[(est_state_df['timestamp']-timestamp).abs().argmin()]
    return closest[theta_key]

def read_and_plot_data(csv_path, plot_data=False):
    # read in the sensor values as a CSV file
    dist_df = pd.read_csv(os.path.join(csv_path, 'Distance.csv')).set_index('timestamp')
    rpm_df = pd.read_csv(os.path.join(csv_path, 'Rpm.csv'))
    # acc_df = pd.read_csv(os.path.join(csv_path, 'Acceleration.csv'))
    est_state_df = pd.read_csv(os.path.join(csv_path, 'EstimatedState.csv'))
    depth_df = pd.read_csv(os.path.join(csv_path, 'Depth.csv'))

    entity_key = [key for key in dist_df.keys() if 'entity' in key][0]
    value_key = [key for key in dist_df.keys() if 'value (m)' in key][0]
    rpm_value_key = [key for key  in rpm_df.keys() if 'value (rpm)' in key][0]
    depth_val_key = [key for key in depth_df.keys() if 'value (m)' in key][0]
    depth_entity_key = [key for key in depth_df.keys() if 'entity' in key][0]
    validity_key = [key for key in dist_df.keys() if 'validity (enumerated)' in key][0]
    theta_key = [key for key in est_state_df.keys() if 'theta' in key][0]

    est_state_df[theta_key] = est_state_df[theta_key].apply(lambda pitch_rad: math.degrees(pitch_rad))

    # print('Entity key: "', entity_key, '"', sep='')
    # print('Value key: "', value_key, '"', sep='')
    # print('Validity key: "', validity_key, '"', sep='')
    # print('Theta key: "', theta_key, '"', sep='')

    # preprocess the data such that all the sensor values that belong to the same timestamp will be in the same entry of the dataframe
    # sensor_data = pd.DataFrame(columns=['timestamp','DVL-0','DVL-1','DVL-2','DVL-3','DVL-filtered','Echo', 'validity', 'rpm', 'depth'])
    index = -1
    sensor_vals = {}
    collected_sensor_vals = []
    for timestamp, data in dist_df.iterrows():
        if index == -1:
            index = timestamp
            sensor_vals['timestamp'] = timestamp

        if timestamp != index:
            sensor_vals['rpm'] = get_closest_rpm(timestamp, rpm_value_key, rpm_df)
            sensor_vals['depth'] = get_closest_depth(timestamp, depth_val_key, depth_entity_key, depth_df)
            sensor_vals['theta'] = get_closest_theta(timestamp, theta_key, est_state_df)

            # sensor_data = pd.concat([sensor_data, pd.DataFrame.from_records([sensor_vals])])
            collected_sensor_vals.append(sensor_vals.copy())
            # print(sensor_vals)
            # sensor_data = sensor_data.append(sensor_vals, ignore_index=True)
            index = timestamp
            sensor_vals['timestamp'] = timestamp
            
        if 'DVL - Beam 0' in data[entity_key]:
            sensor_vals['DVL-0'] = data[value_key]
        elif 'DVL - Beam 1' in data[entity_key]:
            sensor_vals['DVL-1'] = data[value_key]
        elif 'DVL - Beam 2' in data[entity_key]:
            sensor_vals['DVL-2'] = data[value_key]
        elif 'DVL - Beam 3' in data[entity_key]:
            sensor_vals['DVL-3'] = data[value_key]
        elif 'DVL Filtered' in data[entity_key] or 'Altimeter' in data[entity_key]:
            sensor_vals['DVL-filtered'] = data[value_key]
            sensor_vals['validity'] = 'INVALID' if 'INVALID' in data[validity_key] else 'VALID'
        elif 'Echo Sounder' in data[entity_key]:
            sensor_vals['Echo'] = data[value_key]

    sensor_data = pd.DataFrame.from_records(collected_sensor_vals)

    if 'Echo' not in sensor_data.keys():
        return pd.DataFrame(columns=['timestamp','depth','DVL-0','DVL-1','DVL-2','DVL-3','DVL-filtered','Echo','validity','rpm','theta'])

    # filter out entries where at least one sensor does not have output value
    sensor_data = sensor_data[sensor_data['Echo'].notnull()]
    # print(sensor_data)

    if plot_data:
        plot_input_data(sensor_data)

    return sensor_data

def label_data_automatically(sensor_data, plot_data=False):
    global k

    if sensor_data.empty:
        sensor_data['Label'] = []
        return sensor_data

    # first using the 3 states label: AVOIDING, CLIMBING or TRACKING
    # if the forward distance < 28 (for a longer time) go to CLIMBING
    # if the bottom distance < 5 (for a longer time) go to AVOIDING
    # otherwise go to TRACKING

    # ORIGINAL parameters
    # forward_threshold = 15
    # forward_hard_threshold = 8
    # bottom_threshold = 1.2
    # depth_threshold = 0.5

    # attack_time_forward = 20
    # release_time_forward = -10

    # attack_time_forward_hard = 20
    # release_time_forward_hard = -5

    # attack_time_bottom = 5
    # release_time_bottom = -10

    # attack_time_depth = 10
    # release_time_depth = -10

    # refined parameters
    forward_threshold = 26
    forward_hard_threshold = 10
    bottom_threshold = 1.8
    depth_threshold = 0.5

    attack_time_forward = 15
    release_time_forward = -15

    attack_time_forward_hard = 10
    release_time_forward_hard = -10

    attack_time_bottom = 5
    release_time_bottom = -10

    attack_time_depth = 10
    release_time_depth = -10

    forward_vals = []
    forward_hard_vals = []
    bottom_vals = []
    depth_vals = []

    labels = []

    forward_close = False
    forward_hard_close = False
    bottom_close = False
    surfaced = True if sensor_data.iloc[0]['depth'] <= 0.5 else False
    forward_val = 0
    forward_hard_val = 0
    bottom_val = 0
    depth_val = 0
    for index, row in sensor_data.iterrows():
        forward_vals.append(forward_val)
        forward_hard_vals.append(forward_hard_val)
        bottom_vals.append(bottom_val)
        depth_vals.append(depth_val)

        timestamp = row['timestamp']
        dvl_val = row['DVL-filtered']
        echo_val = row['Echo']
        validity = row['validity']
        depth = row['depth']
        # print(timestamp, dvl_val, echo_val)

        if echo_val <= forward_threshold:
            if forward_val < 0:
                forward_val = 0
            else:
                forward_val = min(forward_val + 1, attack_time_forward)
            
            if forward_val == attack_time_forward:
                forward_close = True
        else:
            if  forward_val > 0:
                forward_val = 0
            else:
                forward_val = max(forward_val - 1, release_time_forward)
            
            if forward_val == release_time_forward:
                forward_close = False

        if echo_val <= forward_hard_threshold:
            if forward_hard_val < 0:
                forward_hard_val = 0
            else:
                forward_hard_val = min(forward_hard_val + 1, attack_time_forward_hard)
            
            if forward_hard_val == attack_time_forward_hard:
                forward_hard_close = True
        else:
            if  forward_hard_val > 0:
                forward_hard_val = 0
            else:
                forward_hard_val = max(forward_hard_val - 1, release_time_forward_hard)
            
            if forward_hard_val == release_time_forward_hard:
                forward_hard_close = False

        if depth <= depth_threshold:
            if depth_val < 0:
                depth_val = 0
            else:
                depth_val = min(depth_val + 1, attack_time_depth)
            
            if depth_val == attack_time_depth:
                surfaced = True
        else:
            if  depth_val > 0:
                depth_val = 0
            else:
                depth_val = max(depth_val - 1, release_time_depth)
            
            if depth_val == release_time_depth:
                surfaced = False

        if validity == 'VALID':
            if dvl_val <= bottom_threshold:
                if bottom_val < 0:
                    bottom_val = 0
                else:
                    bottom_val = min(bottom_val + 1, attack_time_bottom)
                
                if bottom_val == attack_time_bottom:
                    bottom_close = True
            else:
                if bottom_val > 0:
                    bottom_val = 0
                else:
                    bottom_val = max(bottom_val - 1, release_time_bottom)
                
                if bottom_val == release_time_bottom:
                    bottom_close = False

        if surfaced:
            labels.append('SURFACED')
        else:
            if bottom_close or forward_hard_close:
                labels.append('AVOIDING')
            else:
                if forward_close:
                    labels.append('CLIMBING')
                else:
                    labels.append('TRACKING')
        
    # afterwards if there is stage with frequent changes between states re-label it with UNSTEADY
    # if the different is bigger than a threshold using the same attack-time release-time algorithm should reveal where the data is unstable

    echo_diff_std_threshold = 6.50
    dvl_diff_std_threshold = 1.00

    timestamps = []
    echo_diff_stds = []
    dvl_diff_stds = []

    window_size = TIME_WINDOW_NUM_SAMPLE
    for i in range(window_size, len(sensor_data) - 1):
        prev_vals = sensor_data.iloc[i - window_size:i]
        echo_vals = prev_vals['Echo']
        dvl_vals = prev_vals['DVL-filtered'].loc[prev_vals['validity']  == 'VALID']  
        # dvl_vals = prev_vals['DVL-filtered'] # for the paper consider also consider the original!!

        echo_diff = np.diff(echo_vals)
        dvl_diff = np.diff(dvl_vals)

        echo_diff_abs = np.abs(echo_diff)
        dvl_diff_abs = np.abs(dvl_diff)

        echo_diff_std = np.std(echo_diff_abs)
        dvl_diff_std = np.std(dvl_diff_abs)

        if echo_diff_std > echo_diff_std_threshold or dvl_diff_std > dvl_diff_std_threshold:
            labels[i] = 'UNSTEADY'

        timestamps.append(prev_vals['timestamp'].tolist()[-1])
        echo_diff_stds.append(echo_diff_std)
        dvl_diff_stds.append(dvl_diff_std)


    sensor_data['Label'] = labels[(TIME_WINDOW_NUM_SAMPLE // 2):] + [labels[-1]] * (TIME_WINDOW_NUM_SAMPLE // 2)
    # print(sensor_data)

    # plt.rcParams['axes.titley'] = -0.5
    # plt.rcParams['axes.titlepad'] = 0  # pad is in points...

    if plot_data:
        fig, axs = plt.subplots(4, 2, sharex=True)
        axs[0][0].plot(sensor_data['timestamp'], sensor_data['Echo'])
        axs[0][0].set_title('Echo sounder value (m)')
        axs[1][0].axhline(y = attack_time_forward, color = 'r', linestyle = '-')
        axs[1][0].axhline(y = release_time_forward, color = 'g', linestyle = '-')
        axs[1][0].plot(sensor_data['timestamp'], forward_vals)
        axs[1][0].set_title('CLIMBING state trigger based on Echo sounder')
        axs[2][0].axhline(y = attack_time_forward_hard, color = 'r', linestyle = '-')
        axs[2][0].axhline(y = release_time_forward_hard, color = 'g', linestyle = '-')
        axs[2][0].plot(sensor_data['timestamp'], forward_hard_vals)
        axs[2][0].set_title('AVOIDING state trigger based on Echo sounder')
        axs[0][1].plot(sensor_data['timestamp'], sensor_data['rpm'])
        axs[0][1].set_title('Rotor speed (rpm)')
        axs[1][1].plot(sensor_data['timestamp'], sensor_data['DVL-filtered'], label='DVL')
        axs[1][1].plot(sensor_data['timestamp'], sensor_data['depth'], label='Depth')
        axs[1][1].legend(loc='upper right')
        axs[1][1].set_title('DVL-filtered and depth value (m)')
        axs[2][1].axhline(y = attack_time_bottom, color = 'r', linestyle = '-')
        axs[2][1].axhline(y = release_time_bottom, color = 'g', linestyle = '-')
        axs[2][1].plot(sensor_data['timestamp'], bottom_vals)
        axs[2][1].set_title('AVOIDING state trigger based on DVL-filtered')
        axs[3][0].axhline(y = echo_diff_std_threshold, color = 'g', linestyle = '-')
        axs[3][0].plot(timestamps, echo_diff_stds)
        axs[3][0].set_title('Echo sounder noise level detector')
        axs[3][1].axhline(y = dvl_diff_std_threshold, color = 'g', linestyle = '-')
        axs[3][1].plot(timestamps, dvl_diff_stds)
        axs[3][1].set_title('DVL-filtered noise level detector')
        # plt.show()

        # fig, axs = plt.subplots(4, 1, sharex=True)
        # axs[0].plot(sensor_data['timestamp'], sensor_data['rpm'])
        # axs[0].set_title('Rotor speed (rpm)')
        # axs[1].plot(sensor_data['timestamp'], sensor_data['Echo'])
        # axs[1].set_title('Echo sounder value (m)')
        # axs[2].axhline(y = attack_time_forward, color = 'r', linestyle = '-')
        # axs[2].axhline(y = release_time_forward, color = 'g', linestyle = '-')
        # axs[2].plot(sensor_data['timestamp'], forward_vals)
        # axs[2].set_title('CLIMBING state trigger based on Echo sounder')
        # axs[3].axhline(y = echo_diff_std_threshold, color = 'g', linestyle = '-')
        # axs[3].plot(timestamps, echo_diff_stds)
        # axs[3].set_title('Echo sounder noise level detector')
        # plt.show()

    # fig.tight_layout()
    # fig.savefig('automatically_labeled_data_' + str(k) + '.png')

    return sensor_data

def prepare_data():
    global k

    print()
    for subdir, dirs, files in os.walk(RAW_CSV_INPUTS_DIR):
        print('Processing:', subdir)
        if set(PROCESSED_CSV_FILES).issubset(set(files)) :  # we have 4 CSV files in each subdir      
            print('All CSV files found.')
            sensor_data_ori = read_and_plot_data(subdir, False)
            # print(sensor_data_ori)

            sensor_data_automatically_labeled = label_data_automatically(sensor_data_ori.copy(), True)
            # print(sensor_data_automatically_labeled)

            plot_labeled_data(sensor_data_automatically_labeled)

            new_subdir = os.path.join('./labeled_CSV_files/', "/".join(subdir.strip("/").split('/')[2:]))
            if not os.path.exists(new_subdir):
                os.makedirs(new_subdir)
            sensor_data_automatically_labeled.to_csv(os.path.join(new_subdir, 'train_data.csv'), index=False)

            k = k + 1
        else:
            print('Not found files:', set(PROCESSED_CSV_FILES).difference(set(files)))
        print('-----------------------------------------', '\n')


if __name__ == '__main__':
    print('Program starting...')

    prepare_data()

    print('Successfully ended.')
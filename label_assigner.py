
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TIMESTEPS_PER_RECORD = 50

COLOR_MAP = {'TRACKING': 'green', 'CLIMBING': 'yellow', 'AVOIDING': 'red', 'UNSTEADY': 'black', 'SURFACED': 'blue'}

class LabelAssigner:

    timestamps = []
    altitude_values = []
    distance_values = []
    depth_values = []
    labels = []

    plotted = False

    def __init__(self,
                 forward_threshold, attack_time_forward, release_time_forward,
                 forward_hard_threshold, attack_time_forward_hard, release_time_forward_hard,
                 bottom_threshold, attack_time_bottom, release_time_bottom,
                 depth_threshold, attack_time_depth, release_time_depth,
                 echo_diff_std_threshold, dvl_diff_std_threshold):
        
        self.forward_threshold = forward_threshold
        self.attack_time_forward = attack_time_forward
        self.release_time_forward = release_time_forward
        self.forward_counter = 0

        self.forward_hard_threshold = forward_hard_threshold
        self.attack_time_forward_hard = attack_time_forward_hard
        self.release_time_forward_hard = release_time_forward_hard
        self.forward_hard_counter = 0

        self.bottom_threshold = bottom_threshold
        self.attack_time_bottom = attack_time_bottom
        self.release_time_bottom = release_time_bottom
        self.bottom_counter = 0

        self.depth_threshold = depth_threshold
        self.attack_time_depth = attack_time_depth
        self.release_time_depth = release_time_depth
        self.depth_counter = 0

        self.echo_diff_std_threshold = echo_diff_std_threshold
        self.dvl_diff_std_threshold = dvl_diff_std_threshold

        self.forward_close = False
        self.forward_hard_close = False
        self.bottom_close = False
        self.surfaced = False
        self.too_noisy = False

    def feed_data(self, timestamp, altitude, distance, depth):
        self.timestamps.append(timestamp)
        self.altitude_values.append(altitude)
        self.distance_values.append(distance)
        self.depth_values.append(depth)

        self.update_counters()

    def update_counters(self):
        # test if the altitude is too low
        if self.altitude_values[-1] <= self.bottom_threshold:
            if self.bottom_counter < 0:
                 self.bottom_counter = 0
            else:
                self.bottom_counter = min(self.bottom_counter + 1, self.attack_time_bottom)

            if self.bottom_counter == self.attack_time_bottom:
                self.bottom_close = True
        else:
            if self.bottom_counter > 0:
                 self.bottom_counter = 0
            else:
                self.bottom_counter = max(self.bottom_counter - 1, self.release_time_bottom)

            if self.bottom_counter == self.release_time_bottom:
                self.bottom_close = False

        # check if the forward distance is short
        if self.distance_values[-1] <= self.forward_threshold:
            if self.forward_counter < 0:
                 self.forward_counter = 0
            else:
                self.forward_counter = min(self.forward_counter + 1, self.attack_time_forward)

            if self.forward_counter == self.attack_time_forward:
                self.forward_close = True
        else:
            if self.forward_counter > 0:
                self.forward_counter = 0
            else:
                self.forward_counter = max(self.forward_counter - 1, self.release_time_forward)

            if self.forward_counter == self.release_time_forward:
                self.forward_close = False

        # check if the forward distance is very short
        if self.distance_values[-1] <= self.forward_hard_threshold:
            if self.forward_hard_counter < 0:
                self.forward_hard_counter = 0
            else:
                self.forward_hard_counter = min(self.forward_hard_counter + 1, self.attack_time_forward_hard)
            
            if self.forward_hard_counter == self.attack_time_forward_hard:
                self.forward_hard_close = True
        else:
            if self.forward_hard_counter > 0:
                self.forward_hard_counter = 0
            else:
                self.forward_hard_counter = max(self.forward_hard_counter - 1, self.release_time_forward_hard)

            if self.forward_hard_counter == self.release_time_forward_hard:
                self.forward_hard_close = False

        # check if the depth is low
        if self.depth_values[-1] <= self.depth_threshold:
            if self.depth_counter < 0:
                self.depth_counter = 0
            else:
                self.depth_counter = min(self.depth_counter + 1, self.attack_time_depth)

            if self.depth_counter == self.attack_time_depth:
                self.surfaced = True
        else:
            if self.depth_counter > 0:
                self.depth_counter = 0
            else:
                self.depth_counter = max(self.depth_counter - 1, self.release_time_depth)

            if self.depth_counter == self.release_time_depth:
                self.surfaced = False

        # check for noisyness
        if len(self.distance_values) > TIMESTEPS_PER_RECORD:
            distances = self.distance_values[-TIMESTEPS_PER_RECORD:]
            dist_diff = np.diff(distances)
            dist_abs_diff = np.abs(dist_diff)
            dist_std_abs_diff = np.std(dist_abs_diff)

            if dist_std_abs_diff >= self.echo_diff_std_threshold:
                self.too_noisy = True
            else:
                self.too_noisy = False

        if len(self.altitude_values) > TIMESTEPS_PER_RECORD:
            altitudes = self.altitude_values[-TIMESTEPS_PER_RECORD:]
            alt_diff = np.diff(altitudes)
            alt_abs_diff = np.abs(alt_diff)
            alt_std_abs_diff = np.std(alt_abs_diff)

            if alt_std_abs_diff >= self.dvl_diff_std_threshold:
                self.too_noisy = True
            else:
                self.too_noisy = False

        if self.too_noisy:
            self.labels.append((self.timestamps[-1], 'UNSTEADY'))
        elif self.surfaced:
            self.labels.append((self.timestamps[-1], 'SURFACED'))
        elif self.bottom_close or self.forward_hard_close:
            self.labels.append((self.timestamps[-1], 'AVOIDING'))
        elif self.forward_close:
            self.labels.append((self.timestamps[-1], 'CLIMBING'))
        else:
            self.labels.append((self.timestamps[-1], 'TRACKING'))
        
        # print(self.labels[-1])

    def get_time_window_labels(self, start_time, end_time):
        return [(timestamp, label) for (timestamp, label) in self.labels if timestamp >= start_time and timestamp <= end_time]

    def do_plot(self):
        # after 5 min plot the labels
        if self.timestamps[-1] - self.timestamps[0] >= 1 * 60 and not self.plotted:
            sensor_data_automatically_labeled = pd.DataFrame(
                list(zip(self.timestamps, self.altitude_values, self.distance_values, self.depth_values, self.labels)),
                columns = ['timestamp', 'DVL-filtered', 'Echo', 'depth', 'Label'])
            self.plotted = True
            self.plot_labeled_data(sensor_data_automatically_labeled)
        else:
            print('Not enough data, currently we have', str(self.timestamps[-1] - self.timestamps[0]), 'seconds')

    def plot_labeled_data(self, sensor_data_automatically_labeled):
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


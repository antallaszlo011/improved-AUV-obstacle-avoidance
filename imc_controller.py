"""
The following example describes how to use the ActorBase class.
When derived from this class announces itself as a CCU, similar to Neptus, and can send and receive IMC messages.
The event loop is based on asyncio. Interactions with asyncio is done through the decorators.
@Subscribe adds a subscriber to a certain IMC message
@Periodic adds a function to be run periodically by the event loop.
"""

import logging
import math
import os
import sys
import time

import imcpy
from imcpy.actors.dynamic import DynamicActor
from imcpy.decorators import Periodic, Subscribe

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, LSTM, AveragePooling1D

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
from scipy.stats import linregress

from label_assigner import LabelAssigner

TIMESTEPS_PER_RECORD = 50  # 10 sec with 0.2 ms frequency
SAMPLE_DIMENSIONS = 4
BUFFER_SIZE = 10000

TRAIN_RATIO = 0.80
OLD_TRAIN_KEEP_RATIO = 1.00

do_plot = False
send_control_commands = True

MODEL_OUTPUTS = ['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY']

COLOR_MAP = {'TRACKING': 'green', 'CLIMBING': 'yellow', 'AVOIDING': 'red', 'UNSTEADY': 'black', 'SURFACED': 'blue'}


class ExampleActor(DynamicActor):
    # with counters we count the number of samples that we have received
    count_depth = 0
    count_altitude = 0
    count_distance = 0

    depth_values = [0.1] * TIMESTEPS_PER_RECORD
    depth_timestamps = [time.time()] * TIMESTEPS_PER_RECORD
    altitude_values = [5.0] * TIMESTEPS_PER_RECORD
    altitude_timestamps = [time.time()] * TIMESTEPS_PER_RECORD
    distance_values = [30.0] * TIMESTEPS_PER_RECORD
    distance_timestamps = [time.time()] * TIMESTEPS_PER_RECORD
    pitch_values = [0.0] * TIMESTEPS_PER_RECORD
    pitch_timestamps = [time.time()] * TIMESTEPS_PER_RECORD

    past_climbings = []

    past_predictions = []
    retrain_data_X = []
    retrain_data_y = []
    retrain_data_t = []
    training_in_progress = False
    plot_shown = False
    on_surface = False
    alt_reached = False
    plot_can_shown = False

    # print(len(depth_values), len(depth_timestamps), len(altitude_values), len(altitude_timestamps), len(distance_values), len(distance_timestamps))

    received = False

    if do_plot:
        # enable interactive mode
        plt.ion()
        
        # creating subplot and figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(depth_timestamps, depth_values)
        line2, = ax.plot(altitude_timestamps, altitude_values)
        line3, = ax.plot(distance_timestamps, distance_values)
        line4, = ax.plot(pitch_timestamps, pitch_values)
        line5, = ax.plot([depth_timestamps[-TIMESTEPS_PER_RECORD], depth_timestamps[-TIMESTEPS_PER_RECORD]], [-15, 30], 'r-.')
        line6, = ax.plot([depth_timestamps[-1], depth_timestamps[-1]], [-15, 30], 'b-.')

        ax.set_xlim(altitude_timestamps[0]-5, altitude_timestamps[-1]+5)
        ax.set_ylim([-15,31])

        # plt.show()

    def __init__(self, target_name):
        """
        Initialize the actor
        :param target_name: The name of the target system
        """
        print('Initializing')

        self.on_surface = False

        super().__init__()
        self.target_name = target_name

        # This list contains the target systems to maintain communications with
        self.heartbeat.append(target_name)

        self.model = self.create_and_load_model()
        print('The model: ')
        self.model.summary()

        self.sensible_label_assigner = LabelAssigner(
                                                        28, 10, -15,
                                                        11, 5, -5,
                                                        2.2, 5, -10,
                                                        0.5, 10, -10,
                                                        6.5, 1.0
                                                    )

        print('Constructor complete')

    @Subscribe(imcpy.EstimatedState)
    def recv_estate(self, msg: imcpy.EstimatedState):
        if self.count_depth - self.count_distance < 3 and self.count_depth - self.count_altitude < 3:
            self.count_depth = self.count_depth + 1

            self.depth_values.append(msg.depth)
            self.depth_timestamps.append(msg.timestamp)
            if len(self.depth_values) > BUFFER_SIZE:
                self.depth_values = self.depth_values[-BUFFER_SIZE:]
                self.depth_timestamps = self.depth_timestamps[-BUFFER_SIZE:]

            pitch = math.degrees(msg.theta)
            self.pitch_values.append(pitch)
            self.pitch_timestamps.append(msg.timestamp)
            if len(self.pitch_values) > BUFFER_SIZE:
                self.pitch_values = self.pitch_values[-BUFFER_SIZE:]
                self.pitch_timestamps = self.pitch_timestamps[-BUFFER_SIZE:]

            if do_plot:
                # updating the value of x and y
                self.line1.set_xdata(self.depth_timestamps)
                self.line1.set_ydata(self.depth_values)

                self.line2.set_xdata(self.altitude_timestamps)
                self.line2.set_ydata(self.altitude_values)

                self.line3.set_xdata(self.distance_timestamps)
                self.line3.set_ydata(self.distance_values)

                self.line4.set_xdata(self.pitch_timestamps)
                self.line4.set_ydata(self.pitch_values)

                self.line5.set_xdata([self.depth_timestamps[-TIMESTEPS_PER_RECORD], self.depth_timestamps[-TIMESTEPS_PER_RECORD]])
                self.line6.set_xdata([self.depth_timestamps[-1], self.depth_timestamps[-1]])

                self.ax.set_xlim(self.altitude_timestamps[0]-5, self.altitude_timestamps[-1]+5)
            
                # re-drawing the figure
                self.fig.canvas.draw()
                
                # to flush the GUI events
                self.fig.canvas.flush_events()
                time.sleep(0.001)


    @Subscribe(imcpy.Distance)
    def recv_distance(self, msg: imcpy.Distance):

        if msg.src_ent == 0x32:
            if self.count_altitude - self.count_depth < 3:
                self.count_altitude = self.count_altitude + 1

                self.altitude_values.append(msg.value)
                self.altitude_timestamps.append(msg.timestamp)
                if len(self.altitude_values) > BUFFER_SIZE:
                    self.altitude_values = self.altitude_values[-BUFFER_SIZE:]
                    self.altitude_timestamps = self.altitude_timestamps[-BUFFER_SIZE:]

        if msg.src_ent == 0x33:
            if self.count_distance - self.count_depth < 3:
                self.count_distance = self.count_distance + 1

                self.distance_values.append(msg.value)
                self.distance_timestamps.append(msg.timestamp)
                if len(self.distance_values) > BUFFER_SIZE:
                    self.distance_values = self.distance_values[-BUFFER_SIZE:]
                    self.distance_timestamps = self.distance_timestamps[-BUFFER_SIZE:]

                self.received = True

                self.sensible_label_assigner.feed_data(self.depth_timestamps[-1], self.altitude_values[-1], self.distance_values[-1], self.depth_values[-1])
                # self.sensible_label_assigner.do_plot()



    @Periodic(0.1)
    def send_state(self):
        if not self.received:
            return

        altitudes = self.altitude_values[-TIMESTEPS_PER_RECORD:]
        distances = self.distance_values[-TIMESTEPS_PER_RECORD:]
        depths = self.depth_values[-TIMESTEPS_PER_RECORD:]
        pitch = self.pitch_values[-TIMESTEPS_PER_RECORD:]
        
        if len(altitudes) == TIMESTEPS_PER_RECORD and len(distances) == TIMESTEPS_PER_RECORD and len(depths) == TIMESTEPS_PER_RECORD:
            zipped_values = [ list(zip(altitudes, distances, depths, pitch)) ]
            input_matrix = np.array(zipped_values)
            pred_y = pd.DataFrame(self.model.predict(input_matrix, verbose=0), columns=['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY'])
            class_label = pred_y.idxmax(axis=1).values[0]
            # print('LSTM prediction:', class_label)

            try:
                # This function resolves the map of connected nodes
                node = self.resolve_node_id(self.target_name)

                # Create a new logbook entry
                log_book_entry = imcpy.LogBookEntry()
                log_book_entry.type = 0
                log_book_entry.timestamp = time.time()
                log_book_entry.context = 'CONTROL_STATE'
                log_book_entry.text = class_label

                # Send the IMC message to the node
                if send_control_commands:
                    self.send(node, log_book_entry)

                    if class_label == 'CLIMBING' or class_label == 'AVOIDING':
                        self.past_climbings.append((self.distance_timestamps[-1], input_matrix[0]))
                
                # save the issued control command 
                self.past_predictions.append((self.distance_timestamps[-1], class_label))

            except KeyError as e:
                # Target system is not connected
                logging.debug('Target system is not connected.')

        else:
            print('Faulty measurment time window sizes')
            print('Len distances:', len(distances))
            print('Len altitudes:', len(altitudes))
            print('Len depths:', len(depths))

    def find_min_and_index(self, arr):
        min_ind = 0
        min_val = arr[0]
        for ind, val in enumerate(arr):
            if val < min_val:
                min_val = val
                min_ind = ind

        return min_ind, min_val
    
    def find_max_and_index(self, arr):
        max_ind = 0
        max_val = arr[0]
        for ind, val in enumerate(arr):
            if val > max_val:
                max_val = val
                max_ind = ind

        return max_ind, max_val
    
    @Periodic(1.0)
    def refine_unnecessary_climbing(self):
        # the other thing to work on to get rid of the "bumps" ... i.e., the unnecessary climbings
        if len(self.altitude_values) < TIMESTEPS_PER_RECORD:
            return

        past_pitch_window = self.pitch_values[-TIMESTEPS_PER_RECORD:]
        past_alt_window = self.altitude_values[-TIMESTEPS_PER_RECORD:]
        past_depth_window = self.depth_values[-TIMESTEPS_PER_RECORD:]

        # check if there is an unnceserray climbing
        min_pitch = min(past_pitch_window)
        max_pitch = max(past_pitch_window)
        # we will need pitch information to check if there was an unncessary climbing (a spike in the pitch, and bump in depth and altitude)
        if min_pitch < -11.5 and max_pitch > 11.5:
            min_depth_ind, min_depth = self.find_min_and_index(past_depth_window)
            max_altitude_ind, min_alt = self.find_max_and_index(past_alt_window)
            if abs(min_depth_ind - 25) <= 5 and abs(max_altitude_ind - 25) <= 5:
                # detecting a bump is good enough
                print('A bump detected !!! time:', str(time.time()))

                # we also need to save all the previous controls and change all the CLIMBING to TRACKING where an unnessary climbing was issued
                bumb_end_time = self.distance_timestamps[-1]
                for timestamp, train_X in self.past_climbings:
                    if timestamp >= bumb_end_time - 15 and timestamp <= bumb_end_time - 5:
                        self.retrain_data_X.append(train_X)
                        self.retrain_data_y.append('TRACKING')
                        self.retrain_data_t.append(timestamp)
                        print('relabel unnecessary climbing')
                        

    @Periodic(1.0)
    def refine_climbing_at_obstacle(self):
        return 
    
        # in this method we should check if the model needs to be retained
        # i.e., if the altitude was too low or too high
        # can we rely on a single altitude measurement? -> no, it may be noise
        # take a window and calculate the avarage? -> could work
        # + apply the previously used signal processing algorithm (implemented in label_assigner.py)

        # first handle if avarage altitude got too low
        # i.e., if avarage altitude is too low, then that data (and also the past data) should be considered as a re-training data
        # also for calculating the good labels use the signal processing algorithm with lowerd thresholds

        if len(self.altitude_values) < TIMESTEPS_PER_RECORD:
            return

        past_altitude_window = self.altitude_values[-TIMESTEPS_PER_RECORD:]
        avg_altitude = np.average(past_altitude_window)
        # print('Average altitude:', avg_altitude)
        
        if avg_altitude <= 2.5:
            # do the retraining
            # relabel the labels from the past ~30 seconds using the more sensitive algorithm
            # train the neural network

            # take the last 50 seconds labeled by the soft labeling method and retrain
            last_labels = self.sensible_label_assigner.get_time_window_labels(self.altitude_timestamps[-1] - 60, self.altitude_timestamps[-1])
            num_labels = len(last_labels)
            timestamps = [timestamp for timestamp, _ in last_labels]
            labels = [label for _, label in last_labels]
            # drop first 25 labels to consider it as future decision
            # labels = labels[25:] + (['UNSTEADY'] * 25)        # the last 25 will be discarded

            altitudes = self.altitude_values[-num_labels:]
            distances = self.distance_values[-num_labels:]
            depths = self.depth_values[-num_labels:]
            pitch = self.pitch_values[-num_labels:]
            # print(len(labels))

            # convert it to training data
            # print('Converting into training data...')
            sensor_data_automatically_labeled = pd.DataFrame(
                list(zip(timestamps, altitudes, distances, depths, pitch, labels)),
                columns = ['timestamp', 'DVL-filtered', 'Echo', 'depth', 'theta', 'Label'])
            # print(sensor_data_automatically_labeled)
            # self.plot_labeled_data(sensor_data_automatically_labeled)

        
            # TODO: problem: now multiple data points may be repeated for training, filter them out using the timestamp

            # now we have the timewindow of recent past (30 seconds)
            # we only work with the first 25 seconds of the timewindow, and it is labeled correspondingly to the last 25 seconds (i.e., shifted by 5)
            # we create this data and collect it into a list of training data in this if branch
            # on the else branch we initiate training  with the data, after finishing we clear the training data list
            # be careful not to initiate multiple trainings simultaneously, until a training is not finished we cannot start a new training
            # compare the final results, somehow visualize the training data? all 30 seconds of data, but coloring only the first 15 seconds
            # only do this when altitude < 2.5


            # TODO: on Friday the problem was here
            # Converting into training data...
            # Reshaping data...
            # ERROR:imcpy.actors.base:Uncaught exception (ValueError) in ExampleActor.refine_climbing_at_obstacle: negative dimensions are not allowed
            # print('Num labels:', str(num_labels))
            # TODO: continue with proofing that everything is correct (labeling of the soft labeler)
            # TODO: introduce pitch as an input

            if num_labels <= TIMESTEPS_PER_RECORD:
                return

            # print('Reshaping data...')
            curr_X = np.empty([num_labels - TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS])
            curr_y = list()
            curr_time = list()
            for i in range(0, num_labels - TIMESTEPS_PER_RECORD):
                train_data = sensor_data_automatically_labeled.iloc[i:i+TIMESTEPS_PER_RECORD][['DVL-filtered', 'Echo', 'depth', 'theta']].values
                train_label = sensor_data_automatically_labeled.iloc[i+TIMESTEPS_PER_RECORD]['Label']
                train_timestamp = sensor_data_automatically_labeled.iloc[i+TIMESTEPS_PER_RECORD]['timestamp']

                curr_X[i] = train_data
                curr_y.append(train_label)
                curr_time.append(train_timestamp)

            # print('Adding to re-train data...')
            for X, y, t in zip(curr_X, curr_y, curr_time):
                if len(self.retrain_data_t) == 0 or not np.any(list(map(lambda time: abs(time - t) < 0.0001, self.retrain_data_t))):     # this is strangely bad
                    # print('Data not close:', t, self.retrain_data_t)
                    self.retrain_data_X.append(X)
                    self.retrain_data_y.append(y)
                    self.retrain_data_t.append(t)
                    print('relabel cliff climbing')
                # else:
                #     print('Data already trained:', t, self.retrain_data_t)


    # TODO: tomorrow test the re-trained model with 500 epochs
    # maybe we need to consider the pitch as input as well
    # maybe it is also necessary to increase the number of LSTM cells 

    @Periodic(1.0)
    def check_if_on_surface(self):
        avg_depth = np.average(self.depth_values[-TIMESTEPS_PER_RECORD:])
        if avg_depth <= 0.25:
            self.on_surface = True
        else:
            self.on_surface = False

    @Periodic(1.0)
    def check_if_alt_reached(self):
        if not self.alt_reached:
            avg_alt = np.average(self.altitude_values[-TIMESTEPS_PER_RECORD:])
            if avg_alt <= 3.5:
                self.alt_reached = True

    def find_vals_in_interval(self, timestamps, values, start_tstamp, end_tstamp):
        result = []
        for tstamp, value in zip(timestamps, values):
            if tstamp >= start_tstamp and tstamp <= end_tstamp:
                result.append(value)

        return np.array(result)

    @Periodic(0.1)
    def refine_unnecessary_climbing_2(self):
        if not self.alt_reached or not self.on_surface:
            return

        for timestamp, data in self.past_climbings:
            depth_vals = data[:, 2]
            pitch_vals = data[:, 3]

            avg_depth = np.average(depth_vals)
            if avg_depth <= 0.5:
                print('change climbing to surfaced')
                self.retrain_data_t.append(timestamp)
                self.retrain_data_X.append(data)
                self.retrain_data_y.append('SURFACED')
                continue

            _, min_pitch = self.find_min_and_index(pitch_vals)
            _, max_pitch = self.find_max_and_index(pitch_vals)
            avg_pitch = np.average(pitch_vals)

            # check if the AUV is diving, if yes CLIMBING not necessary
            if min_pitch <= -7.5 and max_pitch <= -7.5 and avg_pitch <= -7.5:
                print('refine unnecessary climb at diving')
                self.retrain_data_t.append(timestamp)
                self.retrain_data_X.append(data)
                self.retrain_data_y.append('TRACKING')
                continue

            # check the slope of the bethymetry and if not negative then no climbing needed
            # start time is the timestamp of climbing, end time is +25 sec (assuming 1.3 speend and the fact that the echo sounder can detect at distance of 30 meter)
            # ... 30 / 1.3 = ~23
            altitude_after = self.find_vals_in_interval(self.altitude_timestamps, self.altitude_values, timestamp, timestamp + 25)
            depth_after = self.find_vals_in_interval(self.depth_timestamps, self.depth_values, timestamp, timestamp + 25)
            min_len = min(len(altitude_after), len(depth_after))
            altitude_after = altitude_after[:min_len]
            depth_after = depth_after[:min_len]

            bathymetry = np.add(altitude_after, depth_after)

            lin_reg = linregress(list(range(0, min_len)), bathymetry)
            slope_in_degs = math.degrees(math.atan(lin_reg.slope))
            if lin_reg.slope >= 0.001 or math.fabs(slope_in_degs) <= 0.5:
                print('refine unnecessary climb based on bathymetry')
                self.retrain_data_t.append(timestamp)
                self.retrain_data_X.append(data)
                self.retrain_data_y.append('TRACKING')
                continue

            # print('Current bathymetry plot:')
            # print(' - timestamp(s):', timestamp)
            # print(' - slope (ratio):', lin_reg.slope)
            # print(' - slope (in degrees):', slope_in_degs)

            # plt.plot(list(range(0, min_len)), altitude_after)
            # plt.plot(list(range(0, min_len)), depth_after)
            # plt.plot(list(range(0, min_len)), bathymetry)
            # plt.show()

        self.past_climbings = []
        self.plot_can_shown = True



    @Periodic(15.0)
    def retrain_model(self):
        # retrain only on the surface
        if not self.plot_shown or not self.alt_reached or not self.on_surface:
            return

        if len(self.retrain_data_y) > 0 and not self.training_in_progress:
            self.training_in_progress = True

            # do the actual retraining and change the model.. and save (?)
            all_X = self.retrain_data_X     # n x 3 np.array: altitude, distance, depth
            all_y = self.retrain_data_y     # n x 1 list: each element is a control mode
            all_t = self.retrain_data_t     # n x 1 list: each element is the timestamp

            # shuffle the actual retraining data
            perm = np.random.permutation(len(all_y))
            shuff_X = np.array([all_X[ind] for ind in perm])
            shuff_y = [all_y[ind] for ind in perm]
            shuff_t = [all_t[ind] for ind in perm]

            # plot the re-training data here
            # TODO: continue here. plot the data used for re-training ... if incorrect tune the parameters
            # also another problem is that the size of the retraining data is very small (~75) compared to the original training data (~45000) and test data (~15000)

            appended = 0
            for output in MODEL_OUTPUTS:
                if output not in shuff_y:
                    shuff_y.append(output)
                    appended = appended + 1

            # print('Possible labels:', list(set(shuff_y)))
            one_hot_y = pd.get_dummies(shuff_y)
            one_hot_y = one_hot_y.iloc[:-appended]
            print('Shape X:', shuff_X.shape, 'shape y:', one_hot_y.shape)

            
            # load old re-train data
            all_data_X = shuff_X
            all_data_y = one_hot_y
            if os.path.exists('./retrain_data/'):
                for file in os.listdir('./retrain_data/'):
                    if file.endswith('.npy'):
                        print('Using retrain data:', os.path.join('./retrain_data/', file))
                        f = open(os.path.join('./retrain_data/', file), 'rb')
                        re_train_X = np.load(f)
                        re_train_y = np.load(f)
                        all_data_X = np.concatenate([all_data_X, re_train_X])
                        all_data_y = np.concatenate([all_data_y, re_train_y])
            else:
                os.makedirs('./retrain_data/')

            filename = 're_train_data_' + str(int(time.time())) + '.npy'
            with open(os.path.join('./retrain_data/', filename), 'wb') as f:
                np.save(f, shuff_X)
                np.save(f, one_hot_y)


            # load old train data
            if os.path.isfile('./processed_train_data/train_data.npy'):
                f = open('./processed_train_data/train_data.npy', 'rb')
                train_X = np.load(f) 
                train_y = np.load(f) 
                test_X = np.load(f) 
                test_y = np.load(f)

                # drop some of the original training data or weight more the re-train data
                len_train = int(train_y.shape[0] * OLD_TRAIN_KEEP_RATIO)
                train_X = train_X[:len_train]
                train_y = train_y[:len_train]
                len_test = int(test_y.shape[0] * OLD_TRAIN_KEEP_RATIO)
                test_X = test_X[:len_test]
                test_y = test_y[:len_test]

                all_data_X = np.concatenate([all_data_X, train_X, test_X])
                all_data_y = np.concatenate([all_data_y, train_y, test_y])

            perm = np.random.permutation(len(all_data_X))
            shuff_X = all_data_X[perm]
            shuff_y = all_data_y[perm]

            train_size = int(len(shuff_y) * TRAIN_RATIO)

            train_X = shuff_X[:train_size]
            train_y = shuff_y[:train_size]
            test_X = shuff_X[train_size:]
            test_y = shuff_y[train_size:]

            # track if the data used for trianing is labelled correctly (according to the old labeling protocol)

            checkpoint_path = "./model/re-trained_model.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            save_weights_only=True,
                                                            verbose=1)

            model_cpy = tf.keras.models.clone_model(self.model)

            model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model_cpy.build(input_shape=(1, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS))
            model_cpy.fit(
                train_X, train_y, 
                validation_data=(test_X, test_y), 
                epochs=50, 
                batch_size=64, 
                callbacks=[cp_callback])
            
            # Final evaluation of the model
            scores = model_cpy.evaluate(test_X, test_y, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1]*100))
            
            model_cpy.save('re-trained_LSTM_controller.h5')

            self.retrain_data_X = []
            self.retrain_data_y = []
            self.retrain_data_t = []

            self.training_in_progress = False

    def find_closest_pred(self, timestamp, past_preds):
        closest = past_preds.iloc[(past_preds['timestamp']-timestamp).abs().argmin()]
        return closest['label']

    @Periodic(3.0)
    def check_retrain_data(self):
        # print('Alt reached:', self.alt_reached)
        # print('On surface:', self.on_surface)
        # print('Plot showing:', self.plot_showing)
        # print('Len of retrain data:', len(self.retrain_data_y))
        # print()
        
        # if plot is active skip and show this plot only on the surface
        if not self.plot_can_shown:
            return
        
        if self.plot_shown or len(self.retrain_data_y) == 0:
            return
        
        print('Entering data visualization method')
        
        # all_X = self.retrain_data_X.copy()
        # all_y = self.retrain_data_y.copy()
        # all_t = self.retrain_data_t.copy()

        # self.retrain_data_X = []
        # self.retrain_data_y = []
        # self.retrain_data_t = []

        self.plot_shown = True

        min_data_size = min(len(self.altitude_values), len(self.depth_values), len(self.distance_values))

        timestamps = self.distance_timestamps[-min_data_size:]
        altitude_vals = self.altitude_values[-min_data_size:]
        depth_vals = self.depth_values[-min_data_size:]
        distance_vals = self.distance_values[-min_data_size:]
        pitch_vals = self.pitch_values[-min_data_size:]
        past_preds = pd.DataFrame(self.past_predictions, columns=['timestamp', 'label'])
        
        print('Initial data prepared')

        data_df = pd.DataFrame(list(zip(timestamps, altitude_vals, depth_vals, distance_vals, pitch_vals)), columns=['Timestamp', 'Altitude', 'Depth', 'Distance', 'Theta'])
        data_df['Label'] = data_df['Timestamp'].apply(lambda t: self.find_closest_pred(t, past_preds))
        
        print('Added sensor data and predicted labels')

        # assign the refined labels
        data_df['Refined_label'] = data_df['Label']
        for t, y in zip(self.retrain_data_t, self.retrain_data_y):
            closest_index = ((data_df['Timestamp'] - t).abs().argmin())
            data_df.at[closest_index, 'Refined_label'] = y

        print('Added refined labels')

        fig, ax = plt.subplots(4, sharex=True, sharey=True)

        for t, y in zip(self.retrain_data_t, self.retrain_data_y):
            ax[0].vlines(t, -1, 8, linestyles='dashdot')
            ax[1].vlines(t, -1, 8, linestyles='dashdot')
            ax[2].vlines(t, -1, 31, linestyles='dashdot')

        ax[0].scatter(data_df['Timestamp'], data_df['Altitude'], c=[COLOR_MAP[label] for label in data_df['Label'].tolist()])
        ax[1].scatter(data_df['Timestamp'], data_df['Depth'], c=[COLOR_MAP[label] for label in data_df['Label'].tolist()])
        ax[2].scatter(data_df['Timestamp'], data_df['Distance'], c=[COLOR_MAP[label] for label in data_df['Label'].tolist()])
        ax[3].scatter(data_df['Timestamp'], data_df['Theta'], c=[COLOR_MAP[label] for label in data_df['Label'].tolist()])

        ax[0].set_title('Altitude')
        ax[1].set_title('Depth')
        ax[2].set_title('Distance')
        ax[3].set_title('Pitch')

        fig.suptitle('Original labels', fontsize=16)

        fig, ax = plt.subplots(4, sharex=True, sharey=True)

        for t, y in zip(self.retrain_data_t, self.retrain_data_y):
            ax[0].vlines(t, -1, 8, linestyles='dashdot')
            ax[1].vlines(t, -1, 8, linestyles='dashdot')
            ax[2].vlines(t, -1, 31, linestyles='dashdot')

        ax[0].scatter(data_df['Timestamp'], data_df['Altitude'], c=[COLOR_MAP[label] for label in data_df['Refined_label'].tolist()])
        ax[1].scatter(data_df['Timestamp'], data_df['Depth'], c=[COLOR_MAP[label] for label in data_df['Refined_label'].tolist()])
        ax[2].scatter(data_df['Timestamp'], data_df['Distance'], c=[COLOR_MAP[label] for label in data_df['Refined_label'].tolist()])
        ax[3].scatter(data_df['Timestamp'], data_df['Theta'], c=[COLOR_MAP[label] for label in data_df['Refined_label'].tolist()])
        
        ax[0].set_title('Altitude')
        ax[1].set_title('Depth')
        ax[2].set_title('Distance')
        ax[3].set_title('Pitch')

        fig.suptitle('Refined labels', fontsize=16)

        plt.show()



    def create_and_load_model(self):
        # Create a new model instance
        model = tf.keras.models.load_model('LSTM_controller.h5')
        if os.path.exists('./re-trained_LSTM_controller.h5'):
            print('Using the re-trained controller')
            model = tf.keras.models.load_model('./re-trained_LSTM_controller.h5')
        else:
            print('Using the default controller')

        model = tf.keras.models.load_model('re-trained_LSTM_controller_improved2_backup.h5')  # TODO: do not forget to remove

        return model
    
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



if __name__ == '__main__':
    # Setup logging level and console output
    print('Setting up basic configuration')
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    # Create an actor, targeting the lauv-simulator-1 system
    print('Creating actor')
    actor = ExampleActor('lauv-simulator-1')

    # This command starts the asyncio event loop
    print('Running actor')
    actor.run()






























    # @Subscribe(imcpy.EstimatedState)
    # def recv_estate(self, msg: imcpy.EstimatedState):
    #     """
    #     This function is called whenever EstimatedState messages are received
    #     :param msg: Functions decorated with @Subscribe must always have one parameter for the message
    #     :return: None
    #     """

    #     # EstimatedState consists of a reference position (LLH) and a local offset.
    #     # Convert to a single lat/lon coordinate
    #     (lat, lon, hae) = imcpy.coordinates.toWGS84(msg)

    #     print('The estimated position is:', lat, lon, hae)

    # @Periodic(1.0)
    # def run_periodic(self):
    #     """
    #     This function is called every ten seconds. Remember that asyncio (and most of python) is single-threaded.
    #     Doing extensive computations here will halt the event loop. If the UDP buffer fills up this in turn means that
    #     messages will be lost.
    #     :return:
    #     """
    #     logging.info('Periodic function was executed.')

    # @Subscribe(imcpy.Depth)
    # def recv_depth(self, msg: imcpy.Depth):
    #     print('Depth:', msg)

    # @Subscribe(imcpy.Depth)
    # def recv_depth(self, msg: imcpy.Depth):
    #     # print(msg)
    #     if str(msg).splitlines()[1][6] == '0':
    #         # print('DEPTH:', msg.value, 'timestamp:', msg.timestamp)
    #         self.count_depth = self.count_depth + 1
    #         # print('Ratioo: ', float(self.count_depth) / float(self.count_distance))
    #         # print('Ratioo: ', float(self.count_depth) / float(self.count_altitude))
            
    #         if len(self.depth_values) > len(self.distance_values):
    #             self.depth_values[-1] = msg.value
    #         else:
    #             self.depth_values.append(msg.value)
    #         if len(self.depth_values) > 10000:
    #             self.depth_values = self.depth_values[-10000:]

    # @Subscribe(imcpy.Distance)
    # def recv_distance(self, msg: imcpy.Distance):
    #     msg_type = 'DISTANCE' if str(msg).splitlines()[1][6] == '3' else 'ALTITUDE'
    #     # print(msg_type, ':', msg.value, 'timestamp:', msg.timestamp)
    #     if msg_type == 'DISTANCE':
    #         self.count_distance = self.count_distance + 2
    #         self.distance_values.append(msg.value)
    #         # self.distance_values.append(msg.value)
    #         if len(self.distance_values) > 10000:
    #             self.distance_values = self.distance_values[-10000:]
    #     else:
    #         self.count_altitude = self.count_altitude + 2
    #         self.altitude_values.append(msg.value)
    #         # self.altitude_values.append(msg.value)
    #         if len(self.altitude_values) > 10000:
    #             self.altitude_values = self.altitude_values[-10000:]

    #         self.received = True
        

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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, LSTM, AveragePooling1D
from tflite_runtime.interpreter import Interpreter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
# from scipy.stats import linregress

# from label_assigner import LabelAssigner

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

        # self.on_surface = False

        super().__init__()
        self.target_name = target_name

        # This list contains the target systems to maintain communications with
        self.heartbeat.append(target_name)

        self.interpreter, self.input_details, self.output_details = self.create_and_load_model()
        print('The model: ')
        # self.model.summary()
        print(self.input_details)
        print(self.interpreter.get_signature_list())
        print(self.output_details)

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

                # self.sensible_label_assigner.feed_data(self.depth_timestamps[-1], self.altitude_values[-1], self.distance_values[-1], self.depth_values[-1])

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

            self.interpreter.set_tensor(self.input_details[0]['index'], np.array(input_matrix, dtype=np.float32))
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            pred_y = pd.DataFrame(output_data, columns=['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY'])
            class_label = pred_y.idxmax(axis=1).values[0]
            print('LSTM prediction:', class_label)

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

    def evaluate_model(self, interpreter, input_details, output_details):
        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        self.input_shape = input_shape
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(output_data)

        if output_data.shape[1] == 5:
            print('Evaluation test passed')
        else:
            print('Evaluation test failed')


    def create_and_load_model(self):
        # Load the TFLite model and allocate tensors.
        model_path = './re-trained-model_best.tflite'
        interpreter = Interpreter(model_path)
        print('Reading tensorflow lite model file from:', model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        self.evaluate_model(interpreter, input_details, output_details)

        return interpreter, input_details, output_details

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


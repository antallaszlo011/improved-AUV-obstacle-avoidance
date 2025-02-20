import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, LSTM, AveragePooling1D

# INPUT_PATH = './labeled_CSV_files/train_CSV_files/'
# INPUT_PATH = './labeled_CSV_files/test_CSV_files/'
INPUT_PATH = './labeled_CSV_files/recent-mission/'
# INPUT_PATH = './labeled_CSV_files/simulator_CSV_files/'
# INPUT_PATH = './labeled_CSV_files/simulator_curr_CSV_files/'
# INPUT_PATH = './labeled_CSV_files/sim_retrain_CSV_files/'

TIMESTEPS_PER_RECORD = 50  # 1 min with 0.2 ms frequency
SAMPLE_DIMENSIONS = 4

COLOR_MAP = {'TRACKING': 'green', 'CLIMBING': 'yellow', 'AVOIDING': 'red', 'UNSTEADY': 'black', 'SURFACED': 'blue'}

def retrieve_test_data(sensor_data):
    all_X = list()
    all_y = list()

    # conform labels for plotting and training
    sensor_data.loc[(sensor_data['Label'] == 'TRACKING'), 'Color'] = 'green'
    sensor_data.loc[(sensor_data['Label'] == 'CLIMBING'), 'Color'] = 'yellow'
    sensor_data.loc[(sensor_data['Label'] == 'AVOIDING'), 'Color'] = 'red'
    sensor_data.loc[(sensor_data['Label'] == 'UNSTEADY'), 'Color'] = 'black'
    sensor_data.loc[(sensor_data['Label'] == 'SURFACED'), 'Color'] = 'blue'

    sensor_data.loc[(sensor_data['Label'] == 'TRACKING'), 'CLabel'] = 0
    sensor_data.loc[(sensor_data['Label'] == 'CLIMBING'), 'CLabel'] = 1
    sensor_data.loc[(sensor_data['Label'] == 'AVOIDING'), 'CLabel'] = 2
    sensor_data.loc[(sensor_data['Label'] == 'UNSTEADY'), 'CLabel'] = 3
    sensor_data.loc[(sensor_data['Label'] == 'SURFACED'), 'CLabel'] = 4
    # print(sensor_data)

    # convert the data to a training set consisting of time-series with classification labels
    curr_X = np.empty([len(sensor_data) - TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS])
    curr_y = list()
    for i in range(TIMESTEPS_PER_RECORD, len(sensor_data)):
        sample_data = sensor_data.iloc[i - TIMESTEPS_PER_RECORD : i][['DVL-filtered', 'Echo', 'depth', 'theta']].values
        sample_class = sensor_data.iloc[i]['Label']
        # print(sample_data, sample_data.shape, type(sample_data))
        # print(sample_class, sample_class.shape, type(sample_class))

        curr_X[i - TIMESTEPS_PER_RECORD] = sample_data
        curr_y.append(sample_class)

    all_X.append(curr_X)
    all_y.append(curr_y)

    # print(len(all_X), all_X[0].shape, all_X[1].shape, all_X[2].shape)
    # print(len(all_y), len(all_y[0]), len(all_y[1]), len(all_y[2]))
            
    data_X = np.concatenate(all_X, axis=0)
    data_y = [y for part_y in all_y for y in part_y]
    # print(data_X.shape)
    # print(len(data_y))

    one_hot_y = pd.get_dummies(data_y)
    for ind, category in enumerate(['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY']):
        if category not in one_hot_y.keys():
            one_hot_y.insert(loc=ind, column=category, value=[0]*len(one_hot_y))
    
    return (data_X, one_hot_y)

def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(LSTM(8))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(1, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS))
    print(model.summary())

    return model

def create_and_load_model():
    # checkpoint_path = "./model/trained_model.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # latest = tf.train.latest_checkpoint(checkpoint_dir)

    # # Create a new model instance
    # model = create_model()

    # # Load the previously saved weights
    # model.load_weights(latest)
    
    # model = tf.keras.models.load_model('re-trained_LSTM_controller.h5')
    # model = tf.keras.models.load_model('LSTM_controller.h5')
    model = tf.keras.models.load_model('re-trained_LSTM_controller_best.h5')    

    return model

def evaluate_model(test_X, test_y, model):
    # Final evaluation of the model
    scores = model.evaluate(test_X, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    pred_y = pd.DataFrame(model.predict(test_X), columns=['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY'])
    # print(pred_y)
    # print(pred_y.shape)

    return (test_X, pred_y)

def get_colors_from_one_hots(one_hot_vectors):
    class_labels = one_hot_vectors.idxmax(axis=1)
    return [COLOR_MAP[label] for label in class_labels]

def plot_data(test_X, test_y, pred_X, pred_y):
    fig, ax = plt.subplots(2, sharex=True, sharey=True)

    # plot test data
    # ax[0].scatter(range(0, TIMESTEPS_PER_RECORD), test_X[0, :, 5], c=get_colors_from_one_hots(test_y.head(1)))
    ax[0].scatter(range(TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD + len(test_X)), test_X[:, -1, 1], c=get_colors_from_one_hots(test_y))
    ax[0].set_ylabel('Echo sounder value (m)')
    ax[0].set_xlabel('Timestamp (s)')
    ax[0].set_title('Original training data')

    # plot predictions
    # ax[1].scatter(range(0, TIMESTEPS_PER_RECORD), pred_X[0, :, 5], c=get_colors_from_one_hots(pred_y.head(1)))
    ax[1].scatter(range(TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD + len(pred_X)), pred_X[:, -1, 1], c=get_colors_from_one_hots(pred_y))
    ax[1].set_ylabel('Echo sounder value (m)')
    ax[1].set_xlabel('Timestamp (s)')
    ax[1].set_title('Predictions made by the neural network')

    handles, labels = plt.gca().get_legend_handles_labels()

    line1 = Line2D([0], [0], linewidth=5, label='TRACKING', color='green')
    line2 = Line2D([0], [0], linewidth=5, label='CLIMBING', color='yellow')
    line3 = Line2D([0], [0], linewidth=5, label='AVOIDING', color='red')
    line4 = Line2D([0], [0], linewidth=5, label='UNSTEADY', color='black')
    line5 = Line2D([0], [0], linewidth=5, label='SURFACED', color='blue')

    handles.extend([line1, line2, line3, line4, line5])

    fig.legend(handles=handles, loc='upper right')
    plt.show()  


if __name__ == '__main__':
    print('Program started...')

    model = create_and_load_model()
    for subdir, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            print('Processing:', os.path.join(subdir, file), '...')
            sensor_data = pd.read_csv(os.path.join(subdir, file))

            if sensor_data.empty:
                print('DataFrame is empty, skipping')
                continue

            test_X, test_y = retrieve_test_data(sensor_data)
            pred_X, pred_y = evaluate_model(test_X, test_y, model)
            plot_data(test_X, test_y, pred_X, pred_y)

    print('Successfully ended.')
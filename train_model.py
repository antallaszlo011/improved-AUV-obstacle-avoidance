import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, LSTM, AveragePooling1D

INPUT_PATH = './labeled_CSV_files/train_CSV_files/'
TIMESTEPS_PER_RECORD = 50  # 10 sec with 0.2 ms frequency
# TRAIN_SIZE = 4000  # it can be more if the dataset is large enough
TRAIN_RATIO = 0.80
SAMPLE_DIMENSIONS = 4

COLOR_MAP = {'TRACKING': 'green', 'CLIMBING': 'yellow', 'AVOIDING': 'red', 'UNSTEADY': 'black', 'SURFACED': 'blue'}


def retrieve_train_data():
    all_X = list()
    all_y = list()
    for subdir, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            print('Processing:', os.path.join(subdir, file), '...')
            sensor_data = pd.read_csv(os.path.join(subdir, file))

            if sensor_data.empty:
                print('DataFrame is empty, skipping')
                continue

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

    perm = np.random.permutation(len(data_y))
    shuff_X = data_X[perm]
    shuff_y = [data_y[ind] for ind in perm]
    # print(shuff_X.shape)
    # print(len(shuff_y))

    one_hot_y = pd.get_dummies(shuff_y)
    
    train_size = int(len(shuff_y) * TRAIN_RATIO)

    # for i in range(len(data_y)):
    #     plt.figure()
    #     plt.plot(shuff_X[i, :, 5])
    #     plt.plot(shuff_X[i, :, 4])
    #     print('========================================')
    #     print(shuff_y[i])
    #     print(one_hot_y.iloc[i])
    #     print('========================================')
    #     plt.show()

    return (shuff_X[:train_size], one_hot_y.iloc[:train_size], shuff_X[train_size:], one_hot_y.iloc[train_size:])

def train_and_evaluate_model(train_X, train_y, test_X, test_y):
    checkpoint_path = "./model/trained_model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(LSTM(8))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(1, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS))
    print(model.summary())

    # model.fit(np.concatenate([train_X, test_X], axis=0), np.concatenate([train_y, test_y], axis=0), validation_data=(test_X, test_y), epochs=10, batch_size=64)
    model.fit(
        train_X, train_y, 
        validation_data=(test_X, test_y), 
        epochs=100, 
        batch_size=64, 
        callbacks=[cp_callback])


    # Final evaluation of the model
    scores = model.evaluate(test_X, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    pred_y = pd.DataFrame(model.predict(test_X), columns=['AVOIDING', 'CLIMBING', 'SURFACED', 'TRACKING', 'UNSTEADY'])
    # print(pred_y)
    # print(pred_y.shape)


    model.save('LSTM_controller.h5')

    return (test_X, pred_y)

def get_colors_from_one_hots(one_hot_vectors):
    class_labels = one_hot_vectors.idxmax(axis=1)
    return [COLOR_MAP[label] for label in class_labels]

def plot_data(train_X, train_y, test_X, test_y, pred_X, pred_y):
    fig, ax = plt.subplots(4, sharex=True, sharey=True)

    # all data
    ax[0].scatter(range(0, TIMESTEPS_PER_RECORD), train_X[0, :, 5], c=get_colors_from_one_hots(train_y.head(1)))
    ax[0].scatter(range(TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD + len(train_X)), train_X[:, -1, 5], c=get_colors_from_one_hots(train_y))    
    ax[0].scatter(range(TIMESTEPS_PER_RECORD + len(train_X), TIMESTEPS_PER_RECORD + len(train_X) + len(test_X)), test_X[:, -1, 5], c=get_colors_from_one_hots(test_y)) 

    # train data
    ax[1].scatter(range(0, TIMESTEPS_PER_RECORD), train_X[0, :, 5], c=get_colors_from_one_hots(train_y.head(1)))
    ax[1].scatter(range(TIMESTEPS_PER_RECORD, TIMESTEPS_PER_RECORD + len(train_X)), train_X[:, -1, 5], c=get_colors_from_one_hots(train_y)) 

    # test data
    ax[2].scatter(range(TIMESTEPS_PER_RECORD + len(train_X), TIMESTEPS_PER_RECORD + len(train_X) + len(test_X)), test_X[:, -1, 5], c=get_colors_from_one_hots(test_y)) 

    # pred data
    ax[3].scatter(range(TIMESTEPS_PER_RECORD + len(train_X), TIMESTEPS_PER_RECORD + len(train_X) + len(pred_X)), pred_X[:, -1, 5], c=get_colors_from_one_hots(pred_y)) 

    plt.show()  

def save_train_data(train_X, train_y, test_X, test_y):
    with open('processed_train_data/train_data.npy', 'wb') as f:
        np.save(f, train_X)
        np.save(f, train_y)
        np.save(f, test_X)
        np.save(f, test_y)

if __name__ == '__main__':
    print('Program started...')

    train_X, train_y, test_X, test_y = retrieve_train_data()
    # print(train_X, train_y, test_X, test_y)

    save_train_data(train_X, train_y, test_X, test_y)

    pred_X, pred_y = train_and_evaluate_model(train_X, train_y, test_X, test_y)
    # plot_data(train_X, train_y, test_X, test_y, pred_X, pred_y)

    print('Successfully ended.')
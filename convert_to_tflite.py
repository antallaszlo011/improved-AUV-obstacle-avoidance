import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, LSTM, AveragePooling1D

import os

import numpy as np

TIMESTEPS_PER_RECORD = 50  # 1 min with 0.2 ms frequency
SAMPLE_DIMENSIONS = 4

def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(LSTM(8))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(1, TIMESTEPS_PER_RECORD, SAMPLE_DIMENSIONS))
    model.summary()

    return model

def create_and_load_model():
    checkpoint_path = "./model/trained_model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Create a new model instance
    model = create_model()

    # Load the previously saved weights
    model.load_weights(latest)
    
    # model = tf.keras.models.load_model('re-trained_LSTM_controller.h5')
    # model = tf.keras.models.load_model('LSTM_controller.h5')
    # model = tf.keras.models.load_model('re-trained_LSTM_controller_best.h5')    

    return model

print('<======================================>')

# model = create_and_load_model()
model = tf.keras.models.load_model('re-trained_LSTM_controller_best.h5')  

print('<======================================>')

save_model_dir = './final_model'
model.save(save_model_dir)

print('<======================================>')

# print(help(tf.lite.TFLiteConverter))

# converter = tf.lite.TFLiteConverter.from_saved_model('./re-trained_LSTM_controller_best.h5') # path to the SavedModel directory
# tflite_model = converter.convert()

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir) # path to the SavedModel directory
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory


converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

# Save the model.
with open('re-trained-model_best.tflite', 'wb') as f:
  f.write(tflite_model)

print('<======================================>')

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="re-trained-model_best.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


import tensorflow as tf

from keras.models import load_model

# Load and convert the model
model = load_model('model.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Test converted
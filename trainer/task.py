import argparse
import logging.config
import os
import time
import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from tensorflow.keras import callbacks
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from . import __version__

LOGGER = logging.getLogger()
VERSION = __version__

def _download_data():
    LOGGER.info("Downloading data...")
    train, test = datasets.mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    return x_train, y_train, x_test, y_test

def _preprocess_data(x,y,type_of_model):
    
    x = x / 255.0 
    y = utils.to_categorical(y)
    if type_of_model == 'ConvolutionalNN':
        x = x.reshape(-1, 28, 28,1)
    return x,y

def _build_model():
    m = models.Sequential()

    m.add(layers.Input((28,28), name='my_input_layer'))

    m.add(layers.Flatten())

    m.add(layers.Dense(128, activation=activations.relu))

    m.add(layers.Dense(64, activation=activations.relu))

    m.add(layers.Dense(32, activation=activations.relu))

    m.add(layers.Dense(10, activation=activations.softmax))

    return m

def _build_model_CNN(input_shape):
    input_img = Input(shape=input_shape)  
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    y = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    y = MaxPooling2D((2, 2), padding='same')(y)
    y = Conv2D(16, (3, 3), activation='relu', padding='same')(y)
    y = MaxPooling2D((2, 2), padding='same')(y)
    y = Conv2D(16, (3, 3), activation='relu', padding='same')(y)
    y = Dropout(0.4)(y)
    z = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    z = MaxPooling2D((2, 2), padding='same')(z)
    z = Conv2D(16, (3, 3), activation='relu', padding='same')(z)
    z = MaxPooling2D((2, 2), padding='same')(z)
    z = Conv2D(16, (3, 3), activation='relu', padding='same')(z)

    out = layers.concatenate([x, y, z])
    out = layers.Flatten()(out)
    out = Dense(16, activation='selu')(out)
    out = Dense(10, activation='softmax')(out)
    
    model_f = Model(inputs=[input_img], outputs=[out])
    return model_f

def train_and_evaluate(batch_size, epochs, job_dir, output_path, is_hypertune,type_of_model):

    # Download the data
    x_train, y_train, x_test, y_test = _download_data()

    # Preprocess the data
    x_train, y_train = _preprocess_data(x_train,y_train,type_of_model)
    x_test, y_test = _preprocess_data(x_test,y_test,type_of_model)

    # Build the model
    if type_of_model == 'ConvolutionalNN':
        input_shape = [28,28,1]
        m = _build_model_CNN(input_shape)
    else:
        m = _build_model()

    m.compile(loss = losses.categorical_crossentropy, optimizer=optimizers.Adam(),metrics=[metrics.categorical_accuracy])

    # Train the model
    logdir = os.path.join(job_dir, "logs/scalars/" + time.strftime("%Y%m%d-%H%M%S"))
    tb_callback = callbacks.TensorBoard(log_dir=logdir)
    m.fit(x_train,y_train, epochs=epochs, batch_size = batch_size,callbacks=[tb_callback])

    # Evaluate the model
    loss_value, accuracy = m.evaluate(x_test,y_test)
    LOGGER.info(" *** LOSS VALUE: %f  ACCURACY %.4f" % (loss_value,accuracy))

    #Communicate the results of the evaluation of the model
    if is_hypertune:
        metric_tag = 'accuracy_metrics'
        eval_path = os.path.join(job_dir, metric_tag) #job_dir es un path unico, gcp se asegura que así sea
        writer = tf.summary.create_file_writer(eval_path)
        with writer.as_default():
            tf.summary.scalar(metric_tag, accuracy, step=epochs)
        writer.flush()
    
    # Save model in TF SavedModel format
    if not is_hypertune:
        model_dir = os.path.join(output_path,VERSION)
        models.save_model(m, model_dir, save_format = 'tf')


def main():
    """Entry point for your module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypertune', action='store_true', help = 'This is a hypertuning job')
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training')
    parser.add_argument('--type-model',  help='Choose between regular NN or CNN')
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI Platform') # opcion obligatoria
    parser.add_argument('--model-output-path', help='Path to write the SaveModel format')

    args = parser.parse_args()

    is_hypertune = args.hypertune
    type_of_model = args.type_model
    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path, is_hypertune,type_of_model)

if __name__ == "__main__":
    main()
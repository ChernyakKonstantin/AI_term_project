from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM, Dense, Input, Flatten, TimeDistributed
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def make_model():
    model_parameters = {'lstm_units': 100, 'n_outputs': 3}
    # Convolutional part
    cnn = Sequential()
    cnn.add(Resizing(128, 128, interpolation='bilinear'))
    cnn.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Conv2D(filters=32, kernel_size=5, strides=1, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Conv2D(filters=64, kernel_size=5, strides=1, activation='relu'))
    cnn.add(MaxPool2D(pool_size=2, strides=2))
    cnn.add(Flatten())
    # Model
    model = Sequential()
    model.add(Input(shape=(3, None, None, 1)))
    model.add(TimeDistributed(cnn))
    model.add(LSTM(
        model_parameters['lstm_units'],
        return_sequences=True,
        dropout=0.2
        )
    )
    model.add(LSTM(model_parameters['lstm_units']))
    model.add(Dense(model_parameters['n_outputs']))
    model.compile(optimizer='adam', loss='mse')
    return model

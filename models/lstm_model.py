from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model():
    model = Sequential([
        LSTM(300, activation='relu', return_sequences=True, input_shape=(10, 1)),
        Dropout(0.2),
        LSTM(300, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(300, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

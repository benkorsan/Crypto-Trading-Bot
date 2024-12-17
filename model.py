
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    look_back = 20
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=1, epochs=10, verbose=1)

    return model, scaler

def predict_next_price(model, scaler, data):
    last_look_back = data[-20:]
    scaled_data = scaler.transform(np.array(last_look_back).reshape(-1, 1))
    scaled_data = scaled_data.reshape(1, -1, 1)

    predicted_price = model.predict(scaled_data)
    return scaler.inverse_transform(predicted_price)[0, 0]

def update_model_with_loss_threshold(historical_data, model, scaler, loss_threshold=0.01):
    """
    Modeli, mevcut kayıp eşiğini aşarsa günceller.
    """
    new_data = np.array(historical_data).reshape(-1, 1)
    new_data_scaled = scaler.transform(new_data)
    
    X_new, y_new = [], []
    look_back = 20
    for i in range(len(new_data_scaled) - look_back - 1):
        X_new.append(new_data_scaled[i:(i + look_back), 0])
        y_new.append(new_data_scaled[i + look_back, 0])

    X_new = np.array(X_new)
    y_new = np.array(y_new)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    current_loss = model.evaluate(X_new, y_new, verbose=0)
    print(f"Current loss: {current_loss}")
    
    if current_loss > loss_threshold:
        model.fit(X_new, y_new, batch_size=1, epochs=5, verbose=1)

    return model


def update_model(historical_data, model, scaler):
    """
    Modeli yeni verilerle günceller.
    """
    new_data = np.array(historical_data).reshape(-1, 1)
    new_data_scaled = scaler.transform(new_data)

    X_new, y_new = [], []
    look_back = 20
    for i in range(len(new_data_scaled) - look_back - 1):
        X_new.append(new_data_scaled[i:(i + look_back), 0])
        y_new.append(new_data_scaled[i + look_back, 0])

    X_new = np.array(X_new)
    y_new = np.array(y_new)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    model.fit(X_new, y_new, batch_size=1, epochs=5, verbose=1)

    return model

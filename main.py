
import os
import time
import requests
import joblib
import numpy as np
import pandas as pd  # Pandas kütüphanesini import et
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from indicators import calculate_rsi  # calculate_macd içeriği burada tanımlı
from dotenv import load_dotenv
from indicators import calculate_macd

# --- Çevresel Değişkenleri Yükle ---
API_KEY = "xxxxxxxxxxxxx"
API_SECRET = "xxxxxxxxxxxxx"
BASE_URL = "https://api.mexc.com"
STOP_LOSS_PERCENT = 0.01
TRAILING_STOP_PERCENT = 0.005
TRADE_PERCENTAGE = 0.1  # Portföyün %10'u ile işlem yapılır

# --- API İsteklerini Güvenli Hale Getirme ---
def safe_request(url, method="GET", params=None, headers=None):
    try:
        if method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method == "POST":
            response = requests.post(url, params=params, headers=headers)
        else:
            raise ValueError("Desteklenmeyen HTTP yöntemi!")
        if response.status_code == 429:
            print("API Hatası: Çok fazla istek gönderildi. Bekleniyor...")
            time.sleep(60)
            return safe_request(url, method, params, headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Hatası: {e}")
        return None

# --- Mexc Historical Data ---
def get_historical_data(symbol, interval, limit=100):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = safe_request(url, params=params)
    if data:
        # Açılış, Yüksek, Düşük, Kapanış fiyatlarını döndür
        return [(float(item[1]), float(item[2]), float(item[3]), float(item[4])) for item in data]
    return []

def prepare_features(historical_data):
    """
    Özellikleri (features) hazırlama.
    
    :param historical_data: Geçmiş fiyat verileri (close prices).
    :return: Özellikler ve kapanış fiyatları.
    """
    # Eğer historical_data her bir öğe bir tuple içeriyorsa,
    # tuple içinde doğru bir şekilde close fiyatını almak için
    # index kullanabiliriz.
    
    # Eğer 'historical_data' listesinde tuple varsa ve her tuple (fiyat, zaman, vb.) içeriyorsa
    closing_prices = [data[0] for data in historical_data]  # Örneğin, 0. indexte fiyat varsa
    
    # RSI ve MACD'yi hesapla
    rsi = calculate_rsi(closing_prices)  # RSI tek bir değer döndürüyor
    macd, signal = calculate_macd(closing_prices)  # MACD ve Signal listeler olarak döner
    
    # Özellikleri liste haline getir
    features = np.array([[rsi, macd[i], macd[i] - signal[i]] for i in range(1, len(macd))])  # 1. indeksten başlayarak özellikleri oluştur
    return features, closing_prices


# --- Model Eğitim ---
def train_model(features, labels):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f"Model doğruluğu: {model.score(X_test, y_test) * 100:.2f}%")
    return model, scaler

# --- Model ve Verileri Kaydetme ---
def save_model(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path="model.pkl", scaler_path="scaler.pkl"):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print("Model veya scaler dosyası bulunamadı, yeniden eğitim yapılacak.")
        return None, None

# --- İşlem Stratejisi ---
def execute_strategy(symbol, interval):
    historical_data = get_historical_data(symbol, interval)
    if not historical_data:
        print("Veri alınamadı.")
        return

    features, closing_prices = prepare_features(historical_data)

    # Modeli yükle veya eğit
    model, scaler = load_model()
    if not model or not scaler:
        model, scaler = train_model(features, closing_prices)
        save_model(model, scaler)

    position_open = False
    entry_price = None
    trailing_stop = None

    while True:
        new_data = get_historical_data(symbol, interval, limit=10)
        if not new_data:
            print("Güncel veri alınamadı.")
            time.sleep(15)
            continue

        # Özellikleri hazırla
        features, closing_prices = prepare_features(new_data)
        current_price = closing_prices[-1]
        scaled_features = scaler.transform([features[-1]])

        # Tahmin al
        prediction = model.predict(scaled_features)[0]
        print(f"Mevcut fiyat: {current_price}, Tahmin: {'Yükseliş' if prediction == 1 else 'Düşüş'}")

        if prediction == 1 and not position_open:
            entry_price = current_price
            trailing_stop = entry_price * (1 - TRAILING_STOP_PERCENT)
            position_open = True
            print(f"Alış yapıldı: {entry_price}")

        elif position_open:
            trailing_stop = max(trailing_stop, current_price * (1 - TRAILING_STOP_PERCENT))
            if current_price <= trailing_stop:
                print(f"Trailing stop devreye girdi: Pozisyon kapatıldı. Fiyat: {current_price}")
                position_open = False
                entry_price = None

        time.sleep(15)

# --- Ana Çalıştırma ---
if __name__ == "__main__":
    symbol = "BTCUSDT"  # Mexc sembolü
    interval = "1m"     # 1 dakikalık veri
    execute_strategy(symbol, interval)

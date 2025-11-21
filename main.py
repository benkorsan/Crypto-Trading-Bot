import requests
import json
import hmac
import hashlib
import time
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal, ROUND_DOWN
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import websocket
import threading

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

# .env dosyasını yükle
load_dotenv()

# --- API Anahtarları ve URL'ler ---
API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY_HERE")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY_HERE")
BASE_FUTURES_PRIVATE_URL = "https://fapi.binance.com"
BASE_FUTURES_PUBLIC_URL = "https://fapi.binance.com"

# --- Genel Bot Ayarları ---
TARGET_COINS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT",
    "BNBUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT"
]  # 10 coin
DATA_INTERVAL = "15m"  # 15 dakikalık mumlar
DATA_LIMIT = 1000
PREDICTION_PERIOD = 1  # Bir sonraki mum
PRICE_CHANGE_THRESHOLD = Decimal('0.001')
LEVERAGE_RATIO = 20  # 20x kaldıraç
TRADE_AMOUNT_USDT = {coin: Decimal('5') for coin in TARGET_COINS}  # Her işlem için 5 USDT
LOOP_INTERVAL_SECONDS = 10  # 10 saniyede bir kontrol
COMMISSION_RATE = Decimal('0')  # 0 komisyon
RISK_PER_TRADE = Decimal('0.001')  # %1 pozisyon başına risk

# API çağrı sayacı ve global değişkenler
api_call_count = 0
withdrawal_orders = {}
real_time_prices = {}  # WebSocket'ten gelen gerçek zamanlı fiyatlar

# --- Yardımcı Fonksiyonlar ---
def generate_signature(api_secret, query_string):
    return hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def get_server_time_futures():
    endpoint = f"{BASE_FUTURES_PUBLIC_URL}/fapi/v1/time"
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()['serverTime']
    except requests.exceptions.RequestException as e:
        logging.error(f"Sunucu zamanı alınırken hata: {e}")
        return None

def check_api_credentials():
    """API anahtarlarını test eder."""
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v2/account"
    server_time = get_server_time_futures()
    if not server_time:
        logging.error("Sunucu zamanı alınamadı, API testi yapılamıyor.")
        return False
    payload = {'timestamp': server_time, 'recvWindow': 10000}
    query_string = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
    signature = generate_signature(SECRET_KEY, query_string)
    headers = {'X-MBX-APIKEY': API_KEY}
    try:
        response = requests.get(endpoint, headers=headers, params=query_string + f"&signature={signature}", timeout=5)
        response.raise_for_status()
        logging.info("API anahtarları doğrulandı.")
        return True
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400 and "Signature for this request is not valid" in response.text:
            logging.error(
                "Geçersiz imza hatası. Lütfen:\n"
                "1. .env dosyasındaki API_KEY ve SECRET_KEY değerlerini kontrol edin (boşluk veya yanlış karakter olmamalı).\n"
                "2. Sunucu saatinin senkronize olduğundan emin olun: 'sudo ntpdate pool.ntp.org' komutunu çalıştırın.\n"
                "3. Binance panelinde API anahtarının futures ve okuma izinlerini kontrol edin."
            )
        elif response.status_code == 401:
            logging.error(
                "API anahtarı veya IP kısıtlaması hatası. Lütfen:\n"
                "1. Binance panelinde API anahtarının futures işlemleri için izinlerini kontrol edin.\n"
                "2. IP kısıtlaması varsa, sunucu IP'sini (69.62.105.30) whitelist'e ekleyin."
            )
        else:
            logging.error(f"API testi sırasında hata: {e}, Yanıt: {response.text}")
        return False

def make_api_request(method, endpoint, headers=None, params=None):
    global api_call_count
    api_call_count += 1
    if api_call_count > 1000:
        logging.warning("API çağrı sınırı yaklaşıyor, 60 saniye bekleniyor...")
        time.sleep(60)
        api_call_count = 0
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 418], backoff_jitter=0.1)
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        response = session.request(method, endpoint, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        weight_used = response.headers.get('X-MBX-USED-WEIGHT-1M', 'Bilinmiyor')
        logging.info(f"API Ağırlık Kullanımı: {weight_used} (Toplam Çağrı: {api_call_count})")
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"API isteği sırasında hata: {e}, Yanıt: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API isteği sırasında bağlantı hatası: {e}")
        return None

def get_kline_data(symbol, interval, limit):
    endpoint = f"{BASE_FUTURES_PUBLIC_URL}/fapi/v1/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': min(limit, 500)}
    try:
        data = make_api_request('GET', endpoint, params=params)
        if not data:
            params['limit'] = 500
            data = make_api_request('GET', endpoint, params=params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[['timestamp', 'open', 'close', 'high', 'low', 'volume']].set_index('timestamp').dropna()
        return df
    except Exception as e:
        logging.error(f"[{symbol}] Kline verisi alınırken hata: {e}")
        return pd.DataFrame()

def calculate_atr(df, period=14):
    """ATR (Average True Range) hesaplar."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

def prepare_data_for_lightgbm(df, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close', 'open', 'high', 'low', 'volume']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        features = scaled_data[i-look_back:i]
        target = 1 if scaled_data[i, 0] > scaled_data[i-1, 0] else 0  # close > önceki close
        X.append(features.flatten())  # 2D array’i düzleştir
        y.append(target)
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_lightgbm_model(X, y, scaler, symbol):
    if X.shape[0] < 10:
        logging.warning(f"[{symbol}] LightGBM için yeterli veri yok.")
        return None
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.01,
        'feature_fraction': 0.9
    }
    model = lgb.train(params, train_data, num_boost_round=200)
    logging.info(f"[{symbol}] LightGBM Modeli eğitildi.")
    return model, scaler

def predict_next_mum(model, scaler, last_data, look_back=60):
    last_sequence = scaler.transform(last_data[['close', 'open', 'high', 'low', 'volume']].values[-look_back:]).flatten().reshape(1, -1)
    prediction = model.predict(last_sequence)[0]
    return 1 if prediction > 0.5 else 0  # 1: Yükseliş, 0: Düşüş

def get_precision_info(symbol, cache={}):
    if symbol in cache:
        return cache[symbol]
    endpoint = f"{BASE_FUTURES_PUBLIC_URL}/fapi/v1/exchangeInfo"
    try:
        response = make_api_request('GET', endpoint)
        if not response:
            raise requests.exceptions.RequestException("exchangeInfo alınamadı")
        for s in response['symbols']:
            if s['symbol'] == symbol:
                price_filter = next(f for f in s['filters'] if f['filterType'] == 'PRICE_FILTER')
                lot_size = next(f for f in s['filters'] if f['filterType'] == 'LOT_SIZE')
                symbol_info = {
                    'price_precision': s['pricePrecision'],
                    'quantity_precision': s['quantityPrecision'],
                    'tick_size': Decimal(price_filter['tickSize']),
                    'min_qty': Decimal(lot_size['minQty']),
                    'max_qty': Decimal(lot_size['maxQty']),
                    'step_size': Decimal(lot_size['stepSize']),
                    'max_leverage': int(s.get('leverage', 20))
                }
                cache[symbol] = symbol_info
                return symbol_info
        default_info = {
            'price_precision': 2, 'quantity_precision': 3, 'tick_size': Decimal('0.01'),
            'min_qty': Decimal('0.001'), 'max_qty': Decimal('1000'), 'step_size': Decimal('0.001'),
            'max_leverage': 20
        }
        cache[symbol] = default_info
        return default_info
    except Exception as e:
        logging.error(f"[{symbol}] Hassasiyet bilgisi alınamadı: {e}")
        return {
            'price_precision': 2, 'quantity_precision': 3, 'tick_size': Decimal('0.01'),
            'min_qty': Decimal('0.001'), 'max_qty': Decimal('1000'), 'step_size': Decimal('0.001'),
            'max_leverage': 20
        }

def adjust_quantity_and_price(quantity, price, symbol_info):
    tick_size = symbol_info['tick_size']
    step_size = symbol_info['step_size']
    min_qty = symbol_info['min_qty']
    max_qty = symbol_info['max_qty']
    adjusted_price = Decimal(str(price)).quantize(tick_size, rounding=ROUND_DOWN) if price else Decimal('0')
    adjusted_quantity = Decimal(str(quantity)).quantize(step_size, rounding=ROUND_DOWN)
    if adjusted_quantity < min_qty:
        adjusted_quantity = min_qty
    if adjusted_quantity > max_qty:
        adjusted_quantity = max_qty
    return float(adjusted_quantity), float(adjusted_price)

def get_account_balance():
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v2/balance"
    server_time = get_server_time_futures()
    if not server_time:
        logging.error("Sunucu zamanı alınamadı, bakiye kontrolü başarısız.")
        return Decimal('0')
    payload = {'timestamp': server_time, 'recvWindow': 10000}
    query_string = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
    signature = generate_signature(SECRET_KEY, query_string)
    headers = {'X-MBX-APIKEY': API_KEY}
    try:
        response = make_api_request('GET', endpoint, headers=headers, params=query_string + f"&signature={signature}")
        if not response:
            logging.error("Hesap bakiyesi alınamadı: Yanıt boş.")
            return Decimal('0')
        usdt_balance = next((b for b in response if b['asset'] == 'USDT'), None)
        if not usdt_balance:
            logging.error("Hesap bakiyesi alınamadı: USDT bulunamadı.")
            return Decimal('0')
        return Decimal(usdt_balance['availableBalance'])
    except requests.exceptions.HTTPError as e:
        logging.error(f"Hesap bakiyesi alınırken HTTP hatası: {e}, Yanıt: {e.response.text}")
        return Decimal('0')
    except Exception as e:
        logging.error(f"Hesap bakiyesi alınırken beklenmeyen hata: {e}")
        return Decimal('0')

def place_order(symbol, side, quantity, leverage, order_type="MARKET", price=None):
    symbol_info = get_precision_info(symbol)
    if leverage > symbol_info['max_leverage']:
        leverage = symbol_info['max_leverage']
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v1/leverage"
    server_time = get_server_time_futures()
    if not server_time:
        return None
    leverage_payload = {'symbol': symbol, 'leverage': int(leverage), 'timestamp': server_time, 'recvWindow': 10000}
    leverage_query = "&".join([f"{k}={v}" for k, v in sorted(leverage_payload.items())])
    make_api_request('POST', endpoint, headers={'X-MBX-APIKEY': API_KEY}, params=leverage_query + f"&signature={generate_signature(SECRET_KEY, leverage_query)}")
    adjusted_quantity, adjusted_price = adjust_quantity_and_price(quantity, price or 0, symbol_info)
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v1/order"
    payload = {'symbol': symbol, 'side': side.upper(), 'type': order_type, 'quantity': adjusted_quantity, 'timestamp': server_time, 'recvWindow': 10000}
    if order_type == "LIMIT" and price:
        payload['price'] = adjusted_price
        payload['timeInForce'] = 'GTC'
    query_string = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
    try:
        response = make_api_request('POST', endpoint, headers={'X-MBX-APIKEY': API_KEY}, params=query_string + f"&signature={generate_signature(SECRET_KEY, query_string)}")
        if response:
            return response
        return None
    except Exception as e:
        logging.error(f"[{symbol}] Emir oluşturulurken hata: {e}")
        return None

def place_stop_loss_order(symbol, position_side, quantity, stop_loss_price):
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v1/order"
    server_time = get_server_time_futures()
    if not server_time:
        return None
    symbol_info = get_precision_info(symbol)
    adjusted_quantity, _ = adjust_quantity_and_price(quantity, 0, symbol_info)
    order_side = "SELL" if position_side == "BUY" else "BUY"
    adjusted_sl_price = Decimal(str(stop_loss_price)).quantize(symbol_info['tick_size'], rounding=ROUND_DOWN)
    payload = {
        'symbol': symbol, 'side': order_side, 'type': 'STOP_MARKET',
        'quantity': adjusted_quantity, 'stopPrice': float(adjusted_sl_price),
        'reduceOnly': 'true', 'timestamp': server_time, 'recvWindow': 10000
    }
    query_string = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
    try:
        response = make_api_request('POST', endpoint, headers={'X-MBX-APIKEY': API_KEY}, params=query_string + f"&signature={generate_signature(SECRET_KEY, query_string)}")
        if response and 'orderId' in response:
            logging.info(f"[{symbol}] Stop-Loss emri yerleştirildi: {response}")
            return response
        return None
    except Exception as e:
        logging.error(f"[{symbol}] Stop-Loss emri oluşturulurken hata: {e}")
        return None

def get_open_positions():
    endpoint = f"{BASE_FUTURES_PRIVATE_URL}/fapi/v2/positionRisk"
    server_time = get_server_time_futures()
    if not server_time:
        return []
    payload = {'timestamp': server_time, 'recvWindow': 10000}
    query_string = "&".join([f"{k}={v}" for k, v in sorted(payload.items())])
    try:
        response = make_api_request('GET', endpoint, headers={'X-MBX-APIKEY': API_KEY}, params=query_string + f"&signature={generate_signature(SECRET_KEY, query_string)}")
        return [p for p in response if float(p['positionAmt']) != 0] if response else []
    except Exception as e:
        logging.error(f"Pozisyonlar alınırken hata: {e}")
        return []

def get_current_price(symbol):
    if symbol in real_time_prices:
        return real_time_prices[symbol]
    endpoint = f"{BASE_FUTURES_PUBLIC_URL}/fapi/v1/ticker/price"
    try:
        response = make_api_request('GET', endpoint, params={'symbol': symbol})
        price = Decimal(response['price']) if response else None
        if price:
            real_time_prices[symbol] = price
        return price
    except Exception as e:
        logging.error(f"[{symbol}] Anlık fiyat alınırken hata: {e}")
        return None

def on_message(ws, message):
    data = json.loads(message)
    if data and isinstance(data, list):
        for item in data:
            if item.get('e') == '24hrTicker' and item.get('s'):
                symbol = item['s']
                price = Decimal(item['c'])
                real_time_prices[symbol] = price
                logging.info(f"[{symbol}] Gerçek zamanlı fiyat: {price}")

def on_error(ws, error):
    logging.error(f"WebSocket hatası: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket bağlantısı kapandı.")

def on_open(ws):
    logging.info("WebSocket bağlantısı açıldı.")

def start_websocket():
    ws_url = f"wss://fstream.binance.com/ws/!miniTicker@arr"
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

def reverse_position(symbol, current_position, current_price):
    pos_side = "BUY" if float(current_position['positionAmt']) > 0 else "SELL"
    pos_quantity = abs(Decimal(current_position['positionAmt']))
    close_side = "SELL" if pos_side == "BUY" else "BUY"
    place_order(symbol, close_side, float(pos_quantity), LEVERAGE_RATIO, "MARKET")
    new_side = "BUY" if pos_side == "SELL" else "SELL"
    trade_volume = (TRADE_AMOUNT_USDT[symbol] * Decimal(str(LEVERAGE_RATIO))) / Decimal(str(current_price))
    symbol_info = get_precision_info(symbol)
    trade_volume, _ = adjust_quantity_and_price(trade_volume, 0, symbol_info)
    place_order(symbol, new_side, float(trade_volume), LEVERAGE_RATIO, "MARKET")
    logging.info(f"[{symbol}] Pozisyon tersine çevrildi: {pos_side} -> {new_side}")

def run_trading_bot():
    if API_KEY == "YOUR_BINANCE_API_KEY_HERE" or SECRET_KEY == "YOUR_BINANCE_SECRET_KEY_HERE":
        logging.error("API anahtarları güncellenmedi. Lütfen .env dosyasını kontrol edin.")
        return
    if not check_api_credentials():
        logging.error("Bot başlatılamadı: API anahtarları doğrulanamadı.")
        return
    logging.info("Gelişmiş AI Tabanlı Futures Botu Başlatılıyor...")
    close_all_positions()
    symbol_info_cache = {coin: get_precision_info(coin) for coin in TARGET_COINS}
    trained_models = {}
    for coin in TARGET_COINS:
        kline_df = get_kline_data(coin, DATA_INTERVAL, DATA_LIMIT)
        if not kline_df.empty:
            X, y, scaler = prepare_data_for_lightgbm(kline_df)
            if X.shape[0] > 0:
                model, scaler = train_lightgbm_model(X, y, scaler, coin)
                if model:
                    trained_models[coin] = (model, scaler)
                    logging.info(f"[{coin}] Model eğitildi.")
    start_websocket()
    while True:
        try:
            balance = get_account_balance()
            if balance <= 0:
                logging.error(f"Yetersiz bakiye: {balance} USDT.")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue
            open_positions = get_open_positions()
            used_margin = sum((abs(Decimal(p['positionAmt'])) * Decimal(p['entryPrice']) / Decimal(p.get('leverage', LEVERAGE_RATIO))) for p in open_positions)
            available_balance = balance - used_margin
            if available_balance < Decimal('0.10'):
                logging.warning(f"Kullanılabilir bakiye yetersiz: {available_balance} USDT. Yeni pozisyon açılmayacak.")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue
            for coin in TARGET_COINS:
                current_price = get_current_price(coin)
                if not current_price:
                    continue
                kline_df = get_kline_data(coin, DATA_INTERVAL, DATA_LIMIT)
                if kline_df.empty or len(kline_df) < 60:
                    continue
                model, scaler = trained_models.get(coin, (None, None))
                if not model:
                    continue
                prediction = predict_next_mum(model, scaler, kline_df)
                current_position = next((p for p in open_positions if p['symbol'] == coin), None)
                symbol_info = symbol_info_cache[coin]
                trade_volume = (TRADE_AMOUNT_USDT[coin] * Decimal(str(LEVERAGE_RATIO))) / Decimal(str(current_price))
                trade_volume = min(trade_volume, Decimal(str(symbol_info['max_qty'])))
                trade_volume, _ = adjust_quantity_and_price(trade_volume, 0, symbol_info)
                required_margin = (Decimal(str(trade_volume)) * current_price) / Decimal(str(LEVERAGE_RATIO))
                if current_position:
                    pos_side = "BUY" if float(current_position['positionAmt']) > 0 else "SELL"
                    pos_entry_price = Decimal(current_position['entryPrice'])
                    pos_quantity = abs(Decimal(current_position['positionAmt']))
                    profit_loss = ((current_price - pos_entry_price) / pos_entry_price - 2 * COMMISSION_RATE) if pos_side == "BUY" else ((pos_entry_price - current_price) / pos_entry_price - 2 * COMMISSION_RATE)
                    profit_percentage = profit_loss * 100
                    logging.info(f"[{coin}] Açık Pozisyon: SIDE={pos_side}, Giriş Fiyatı={pos_entry_price:.4f}, Kar/Zarar={profit_percentage:.2f}%")
                    # ATR tabanlı dinamik stop-loss
                    atr = calculate_atr(kline_df)
                    multiplier = 2
                    stop_loss_price = pos_entry_price - Decimal(str(atr * multiplier)) if pos_side == "BUY" else pos_entry_price + Decimal(str(atr * multiplier))
                    place_stop_loss_order(coin, pos_side, float(pos_quantity), stop_loss_price)
                    logging.info(f"[{coin}] Stop-Loss Güncellendi (ATR tabanlı): {stop_loss_price:.4f}")
                    # Kademeli kâr takibi (trailing stop)
                    close_side = "SELL" if pos_side == "BUY" else "BUY"
                    trailing_stop_price = current_price * Decimal('0.95') if pos_side == "BUY" else current_price * Decimal('1.05')  # %5 geriden takip
                    place_stop_loss_order(coin, pos_side, float(pos_quantity), trailing_stop_price)
                    logging.info(f"[{coin}] Trailing stop güncellendi: {trailing_stop_price:.4f}")
                    # Tersine çevirme sadece tahmin tersine dönerse ve zarar varsa
                    expected_prediction = 1 if pos_side == "BUY" else 0
                    if prediction != expected_prediction and profit_loss < 0:
                        reverse_position(coin, current_position, current_price)
                        new_stop_loss = current_price - Decimal(str(atr * multiplier)) if prediction == 1 else current_price + Decimal(str(atr * multiplier))
                        place_stop_loss_order(coin, "BUY" if prediction == 1 else "SELL", float(trade_volume), new_stop_loss)
                        logging.info(f"[{coin}] Tahmin tersine döndü, Pozisyon tersine çevrildi, Yeni Stop-Loss (ATR tabanlı): {new_stop_loss:.4f}")
                elif available_balance >= required_margin:
                    atr = calculate_atr(kline_df)
                    multiplier = 2
                    if prediction == 1:
                        place_order(coin, "BUY", float(trade_volume), LEVERAGE_RATIO, "MARKET")
                        stop_loss_price = current_price - Decimal(str(atr * multiplier))
                        place_stop_loss_order(coin, "BUY", float(trade_volume), stop_loss_price)
                        logging.info(f"[{coin}] Yeni LONG Pozisyonu Açıldı, Stop-Loss (ATR tabanlı): {stop_loss_price:.4f}")
                    elif prediction == 0:
                        place_order(coin, "SELL", float(trade_volume), LEVERAGE_RATIO, "MARKET")
                        stop_loss_price = current_price + Decimal(str(atr * multiplier))
                        place_stop_loss_order(coin, "SELL", float(trade_volume), stop_loss_price)
                        logging.info(f"[{coin}] Yeni SHORT Pozisyonu Açıldı, Stop-Loss (ATR tabanlı): {stop_loss_price:.4f}")
                else:
                    logging.warning(f"[{coin}] Yeterli marj yok: Gerekli={required_margin:.4f}, Mevcut={available_balance:.4f}")
            time.sleep(LOOP_INTERVAL_SECONDS)
        except Exception as e:
            logging.critical(f"Hata: {e}")
            time.sleep(30)

def close_all_positions():
    open_positions = get_open_positions()
    for pos in open_positions:
        symbol = pos['symbol']
        pos_qty = abs(Decimal(pos['positionAmt']))
        close_side = "SELL" if float(pos['positionAmt']) > 0 else "BUY"
        place_order(symbol, close_side, float(pos_qty), LEVERAGE_RATIO, "MARKET")
        if symbol in withdrawal_orders:
            del withdrawal_orders[symbol]

if __name__ == "__main__":
    run_trading_bot()

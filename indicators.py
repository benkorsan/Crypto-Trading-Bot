
import pandas as pd

def calculate_rsi(prices, period=14):
    """
    RSI (Relative Strength Index) hesaplama.
    
    :param prices: Fiyat verilerinin listesi.
    :param period: RSI hesaplama dönemi (varsayılan: 14).
    :return: Son RSI değeri.
    """
    # Fiyat değişikliklerini hesapla
    delta = pd.Series(prices).diff(1)
    
    # Kazançları ve kayıpları ayır
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Ortalama kazanç ve kayıp
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # RSI hesaplama
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  # Son RSI değerini döndür

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """
    MACD ve Signal line hesaplama.
    
    :param prices: Fiyat verilerinin listesi.
    :param short_period: Kısa dönem EMA için periyot (varsayılan: 12).
    :param long_period: Uzun dönem EMA için periyot (varsayılan: 26).
    :param signal_period: Sinyal çizgisi için periyot (varsayılan: 9).
    :return: MACD ve Signal line değerlerini döndüren listeler.
    """
    short_ema = pd.Series(prices).ewm(span=short_period).mean()
    long_ema = pd.Series(prices).ewm(span=long_period).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period).mean()
    
    return macd.tolist(), signal.tolist()  # MACD ve Signal line'ı liste olarak döndür

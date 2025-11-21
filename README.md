# Binance Futures AI Trading Bot  
**Sürüm:** 1.0 | **Tarih:** 21 Kasım 2025  

> **UYARI & SORUMLULUK REDDİ**  
> Bu yazılım tamamen **eğitim ve araştırma amaçlı** olarak hazırlanmıştır.  
> Gerçek hesapta kullanmanız durumunda **tüm finansal risk size aittir**.  
> Bot, hiçbir koşulda kar garantisi vermez; piyasa koşulları, likidite, slippage, API gecikmeleri ve hatalar nedeniyle **tamamen sermaye kaybı** yaşanabilir.  
> **Yazar, geliştirici veya dağıtıcı hiçbir şekilde maddi/manevi zarardan sorumlu tutulamaz.**  
> Lütfen önce **testnet** veya **çok küçük miktarlar** ile deneyin.

## Özellikler

| Özellik                              | Açıklama                                                                                  |
|--------------------------------------|-------------------------------------------------------------------------------------------|
| **Çoklu Coin Desteği**               | Aynı anda 10 popüler USDT perpetual kontrat (BTC, ETH, ADA, SOL, XRP, BNB, DOGE, DOT, LINK, AVAX) |
| **LightGBM ile AI Tahmini**          | 15 dakikalık mum verileriyle eğitilen ikili sınıflandırma modeli (yükseliş/düşüş)          |
| **Gerçek Zamanlı Fiyat**             | Binance `!miniTicker@arr` WebSocket ile tüm sembollerin anlık fiyat takibi                |
| **ATR Tabanlı Dinamik Stop-Loss**    | Volatiliteye göre otomatik ayarlanan 2×ATR stop-loss                                     |
| **Kademeli Kâr Takibi**              | %5 geriden takip eden trailing-stop (her döngüde yenilenir)                               |
| **Pozisyon Ters Çevirme**            | Tahmin tersine döner ve mevcut pozisyon zarardaysa otomatik ters işlem + yeni SL          |
| **Kaldıraç & Marj Yönetimi**         | 20× sabit kaldıraç, her coin için 5 USDT marj, bakiye kontrolü                            |
| **Hassasiyet & LOT_SIZE Uyumluluğu** | Binance `exchangeInfo` üzerinden tickSize/stepSize/minQty kontrolü                        |
| **API Rate-Limit Koruması**          | Ağırlık (weight) takibi + otomatik bekleme                                               |
| **Detaylı Loglama**                  | Konsol + `trading_bot.log` dosyasına tam zaman damgalı kayıtlar                           |
| **Otomatik Kapanış**                 | Bot başlangıcında tüm açık pozisyonları kapatır (temiz başlangıç)                         |

## Gereksinimler

```txt
# requirements.txt
requests==2.32.3
websocket-client==1.8.0
python-dotenv==1.0.1
pandas==2.2.3
numpy==2.1.2
lightgbm==4.5.0
scikit-learn==1.5.2

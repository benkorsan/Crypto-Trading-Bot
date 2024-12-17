Mexc Algo Trading Botu
Bu proje, Mexc borsasında BTC/USDT gibi kripto çiftlerinde otomatik alım-satım stratejisi gerçekleştiren bir algoritmik işlem botudur. Bot, geçmiş fiyat verilerini kullanarak RSI ve MACD gibi teknik göstergeleri hesaplar, makine öğrenimi modeli ile alım-satım kararları verir ve trailing stop mekanizması ile riski yönetir.

Özellikler
Otomatik Veri Çekme: Mexc API'sinden işlem çiftleri için geçmiş fiyat verilerini toplar.
Teknik Göstergeler:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence) hesaplamaları yapılır.
Makine Öğrenimi:
Logistic Regression kullanarak model eğitimi ve tahmin yapılır.
Model doğruluğu test edilerek iyileştirilir.
Risk Yönetimi:
Stop Loss: Kayıpları sınırlar.
Trailing Stop: Kârı korumak için dinamik stop loss uygular.
Model Kaydetme ve Yükleme:
Eğitim sonrası model ve scaler kaydedilir, sonraki çalıştırmalarda doğrudan yüklenir.
Gerçek Zamanlı Strateji:
1 dakikalık (1m) zaman diliminde gerçek zamanlı tahmin yapar ve karar verir.
Gereksinimler
Projenin çalışabilmesi için aşağıdaki bağımlılıkların yüklü olması gerekmektedir:

Python >= 3.8
Kütüphaneler:
requests
joblib
numpy
pandas
scikit-learn
python-dotenv
Bağımlılıkları yüklemek için aşağıdaki komutu kullanabilirsiniz:

bash
Kodu kopyala
pip install requests joblib numpy pandas scikit-learn python-dotenv
Kurulum
Proje Dosyalarını İndirin:

bash
Kodu kopyala
git clone <proje-repo-linki>
cd <proje-dizini>
Ortam Değişkenlerini Ayarlayın:

.env dosyası oluşturun ve API_KEY, API_SECRET gibi gerekli değişkenleri ayarlayın.
env
Kodu kopyala
API_KEY=
API_SECRET=
BASE_URL=https://api.mexc.com
Projenin Çalıştırılması:

bash
Kodu kopyala
python main.py
Kullanım
Model Eğitimi:
Program, geçmiş veriler ile modeli otomatik olarak eğitir.
Eğer model daha önce eğitilmişse, kaydedilen modeli yükler.
Alım-Satım Stratejisi:
Bot, gerçek zamanlı fiyat verilerini kullanarak Logistic Regression modeli ile yükseliş/düşüş tahmini yapar.
Yükseliş tahmin edildiğinde pozisyon açar ve trailing stop ile pozisyonu yönetir.
Risk Yönetimi:
STOP_LOSS_PERCENT: %1 zarar durumunda pozisyonu kapatır.
TRAILING_STOP_PERCENT: %0.5 kâr durumunda stop loss'u yukarı taşır.
Dosya Yapısı
plaintext
Kodu kopyala
/
|-- main.py                 # Ana çalışma dosyası
|-- indicators.py           # Teknik göstergeler (RSI, MACD) hesaplama
|-- model.pkl               # Kaydedilmiş model dosyası
|-- scaler.pkl              # Kaydedilmiş scaler dosyası
|-- .env                    # Ortam değişkenleri (API_KEY, API_SECRET)
|-- requirements.txt        # Gerekli bağımlılıkların listesi
|-- README.md               # Proje dokümantasyonu
Fonksiyonlar
safe_request
Amaç: API isteklerini güvenli şekilde gönderir ve hata durumlarını yönetir.
Parametreler:
url: İstek URL'si
method: HTTP yöntemi (GET/POST)
Dönüş: JSON formatında API yanıtı.
get_historical_data
Amaç: Mexc borsasından geçmiş fiyat verilerini çeker.
Parametreler:
symbol: İşlem çifti (BTCUSDT)
interval: Zaman dilimi (1m, 5m)
Dönüş: Fiyat verileri.
prepare_features
Amaç: Model için özellikleri hazırlar (RSI, MACD farkları).
Dönüş: Özellikler ve kapanış fiyatları.
train_model
Amaç: Logistic Regression modeli eğitir.
Dönüş: Model ve scaler nesneleri.
execute_strategy
Amaç: Gerçek zamanlı olarak işlem stratejisini uygular.
Parametreler:
symbol: İşlem çifti
interval: Zaman dilimi
Örnek Çalıştırma
Çalıştırdığınızda terminalde aşağıdaki gibi bir çıktı alırsınız:

plaintext
Kodu kopyala
Model doğruluğu: 87.50%
Mevcut fiyat: 27000.5, Tahmin: Yükseliş
Alış yapıldı: 27000.5
Mevcut fiyat: 27050.7, Tahmin: Yükseliş
Trailing stop devreye girdi: Pozisyon kapatıldı. Fiyat: 26980.0
Notlar
Güncelleme Limiti: API isteklerinin limitlerine dikkat edilmiştir. Aşırı istek durumunda 1 dakika bekler.
Özelleştirme: STOP_LOSS_PERCENT, TRAILING_STOP_PERCENT gibi parametreler kod üzerinden değiştirilebilir.
Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.

Bu detaylı README dosyası, projenizin kullanımı, kurulumu ve çalıştırılması konusunda net bir rehber sağlayacaktır.







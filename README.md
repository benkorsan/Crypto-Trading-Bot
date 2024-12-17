
# **Crypto Trading Bot**  
---
**🚀 Yapay Zeka Destekli Otomatik Kripto Ticaret Botu
Bu proje, MEXC borsası üzerinden sağlanan gerçek zamanlı verilerle çalışarak RSI (Relative Strength Index) ve MACD (Moving Average Convergence Divergence) gibi popüler teknik göstergeleri kullanır. Python tabanlı bu sistem, lojistik regresyon modeli ile geçmiş fiyat verilerini analiz ederek piyasanın yönünü tahmin eder ve bu doğrultuda otomatik alım-satım stratejileri uygular.

Temel Özellikler:

Veri Entegrasyonu: MEXC API üzerinden anlık ve geçmiş fiyat verileri çekilir.

Teknik Göstergeler: RSI ve MACD kullanılarak piyasanın aşırı alım/aşırı satım durumları tespit edilir.

Makine Öğrenimi: Lojistik regresyon modeli ile veriler eğitilerek piyasa hareketleri tahmin edilir.

Otomatik Alım-Satım: Tahmin edilen verilere göre uygun alım-satım pozisyonları açılır ve trailing stop-loss stratejisi ile risk yönetimi sağlanır.

Güçlü Performans: Kullanıcı müdahalesine gerek kalmadan piyasa verilerini analiz ederek stratejileri gerçek zamanlı uygular.**
---
---
⚠️ Yasal Uyarı ve Sorumluluk Reddi
Bu proje tamamen eğitim ve araştırma amaçlı geliştirilmiştir. İçerisindeki kod veya algoritmalar yatırım tavsiyesi niteliği taşımamaktadır ve herhangi bir finansal kazanç/kayıp garantisi sağlamaz.
---
Kripto para ticareti yüksek risk içerir. Bu botu kullanarak yapacağınız işlemlerden doğacak tüm sorumluluk kullanıcıya aittir. Finansal kararlarınızı almadan önce kendi araştırmanızı yapmanızı ve bir finans uzmanına danışmanızı öneririz.
---

## **Özellikler**  
- **MEXC API** ile geçmiş piyasa verilerini alma  
- **RSI** ve **MACD** hesaplamaları ile özellikler oluşturma  
- **Logistic Regression** ile model eğitimi  
- **Trailing Stop** ve **Stop Loss** mekanizmaları ile güvenli işlem stratejileri  
- **Model Kaydetme ve Yükleme**: Eğitimli modeli ve ölçekleyiciyi kaydeder, gerektiğinde yeniden kullanır.  

---

## **Kurulum**  

Proje bağımlılıklarını yüklemek için:  

```bash
pip install -r requirements.txt
```

### **Çevresel Değişkenler**  
API anahtarlarını kullanmak için `.env` dosyası oluşturun ve aşağıdaki gibi yapılandırın:  

```
API_KEY=****************
API_SECRET=****************
BASE_URL=https://api.mexc.com
```

---

## **Bağımlılıklar**  

- **Python 3.x**  
- **Requests**: API ile iletişim kurmak için  
- **scikit-learn**: Model eğitimi ve ölçeklendirme  
- **joblib**: Modelin kaydedilip yüklenmesi  
- **pandas**: Veri analizi  
- **numpy**: Matematiksel işlemler  

Proje bağımlılıkları `requirements.txt` dosyasına dahildir.  

---

## **Kod Yapısı**  

- **safe_request**: API isteklerini güvenli hale getirir.  
- **get_historical_data**: Geçmiş piyasa verilerini çeker.  
- **prepare_features**: RSI ve MACD hesaplar ve özellik seti oluşturur.  
- **train_model**: Logistic Regression modeli eğitir.  
- **execute_strategy**: Alım-satım stratejilerini uygular.  
- **save_model / load_model**: Modeli ve ölçekleyiciyi kaydeder/yükler.  

---

## **Kullanım**  

1. Ana dosyayı çalıştırın:  

```bash
python main.py
```

2. **Sembol** ve **zaman aralığı** ayarlamalarını yapabilirsiniz:  

```python
symbol = "BTCUSDT"  # İşlem yapmak istediğiniz sembol  
interval = "1m"     # Zaman aralığı (örneğin: 1m, 5m, 1h)  
```

---

## **İşlem Mantığı**  

- Bot, **RSI** ve **MACD** verilerini kullanarak piyasanın "yükseliş" ya da "düşüş" tahmininde bulunur.  
- Eğer pozitif sinyal algılanırsa (**Yükseliş**):  
   - İşlem açılır (alış yapılır).  
   - **Trailing Stop** mekanizması ile fiyat düşüşü sınırlandırılır.  
- Eğer fiyat belirlenen **stop loss** seviyesini geçerse:  
   - Pozisyon kapatılır.  

---

## **Notlar**  
- Proje, **MEXC** API'si üzerinden çalışmaktadır.  
- Model ve scaler (`model.pkl` ve `scaler.pkl`) otomatik olarak kaydedilir ve bir sonraki çalıştırmada yüklenir.  
- Eğitim verileriniz yetersizse veya eskiyse, model yeniden eğitilir.  

---

## **Geliştirme**  
- Yeni teknik analiz indikatörleri ekleyebilirsiniz.  
- Farklı machine learning algoritmaları deneyebilirsiniz.  
- API istek optimizasyonu yapılabilir.  
---
---
⚠️ Yasal Uyarı
Bu proje bir yatırım tavsiyesi değildir. Finansal işlemlerinizden doğacak kazanç veya kayıplar tamamen sizin sorumluluğunuzdadır. Lütfen finansal kararlarınızı vermeden önce kendi araştırmanızı yapın ve uzman bir danışmana başvurun.
---

## **Lisans**  
Bu proje **MIT Lisansı** ile korunmaktadır.  

---

**Katkıda bulunmak isterseniz pull request açmaktan çekinmeyin! 😊**  

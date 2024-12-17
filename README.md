
# **Crypto Trading Bot**  

Bu proje, **MEXC** borsasından çekilen verilerle RSI (Relative Strength Index) ve MACD (Moving Average Convergence Divergence) gibi teknik göstergeleri kullanarak **logistik regresyon modeli** eğiten ve alım-satım stratejilerini uygulayan bir Python tabanlı kripto ticaret botudur.  

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

## **Lisans**  
Bu proje **MIT Lisansı** ile korunmaktadır.  

---

**Katkıda bulunmak isterseniz pull request açmaktan çekinmeyin! 😊**  

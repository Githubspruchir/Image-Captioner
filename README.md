# 🖼️ Automated Image Captioning with InceptionV3 + LSTM (Flickr8k + Custom Dataset)

A deep learning–powered image captioning system that generates human-like descriptions for images using a CNN-LSTM architecture. Trained on 8,100+ images and 40,500 captions using TensorFlow/Keras, InceptionV3, GloVe embeddings, and beam search decoding.

---

## 🚀 Features

- **CNN Encoder:** Pretrained InceptionV3 extracts 2048-dim visual features
- **LSTM Decoder:** GloVe-initialized LSTM generates captions from image features
- **Beam Search:** Improves caption fluency over greedy decoding
- **BLEU Evaluation:** Achieves BLEU-1 to BLEU-4 scores of 70% to 50%
- **Preprocessing Cache:** Caches image features in 30 mins to save training time
- **Frontend + CLI:** Upload images via web interface or run directly via terminal

---

## 🧠 Tech Stack

- Python 3.9
- TensorFlow / Keras
- InceptionV3 (ImageNet pretrained)
- GloVe Embeddings (200d)
- NLTK (for BLEU score)
- Flask (frontend)
- HTML/CSS (web interface)

---

## 📁 Dataset

- **Flickr8k Dataset**: 8,000+ images with 5 captions each
- **Custom Dataset**: 100+ manually annotated images
- Total: 8,100 images with 40,500 total captions

---

## 🛠 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/image-captioning
cd image-captioning
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Extract features
```bash
python captioning.py --extract
```

### 4. Train the model
```bash
python captioning.py --train
```

### 5. Predict from CLI
```bash
python captioning.py --predict
# Enter image path when prompted
```

### 6. Run Web App
```bash
python app.py
# Visit http://127.0.0.1:5000
```

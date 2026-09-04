# DeepLearning_TranslateWlsal
# WLASL Sign Language Translator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**[Arabic](#arabic) | [English](#english)**

</div>

---

## 🌍 English

### 📖 Overview

The **WLASL Sign Language Translator** is a deep learning-based system that translates American Sign Language (ASL) videos into text. It uses MediaPipe for landmark extraction and a Seq2Seq model with attention mechanism for translation.

### 🎯 Features

- **Video to Text Translation**: Upload ASL videos and get text translation
- **Real-time Processing**: Fast inference using optimized models
- **RESTful API**: Easy integration with other applications
- **Web Interface**: User-friendly interface for testing
- **Batch Processing**: Process multiple videos simultaneously
- **Multi-format Support**: Accepts video files and base64 frames

### 🏗️ Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Input Video │───▶│ MediaPipe │───▶│ Feature │
│ (ASL Sign) │ │ Landmark │ │ Vector │
│ │ │ Extraction │ │ (1662 dims) │
└─────────────────┘ └──────────────────┘ └────────┬────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Output Text │◀───│ Seq2Seq + │◀───│ Encoder │
│ (Translation) │ │ Attention │ │ (Bi-LSTM) │
└─────────────────┘ └──────────────────┘ └─────────────────┘

### 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch, Seq2Seq with Attention |
| **Computer Vision** | OpenCV, MediaPipe |
| **Backend API** | Flask, Flask-CORS |
| **Data Processing** | NumPy, Pandas |
| **Model Deployment** | PyTorch, Pickle |

### 📊 Model Details

- **Input Features**: 1662 dimensions (33 pose × 4 + 468 face × 3 + 21 hand × 3 × 2)
- **Encoder**: 2-layer Bi-LSTM with 512 hidden units
- **Decoder**: 2-layer LSTM with attention mechanism
- **Vocabulary**: Dynamic vocabulary builder
- **Training**: Teacher forcing with 0.5 ratio

### 🚀 Installation

#### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (optional but recommended)

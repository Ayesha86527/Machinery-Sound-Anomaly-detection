# 🔊 Machinery Sound Classification using Transfer Learning

## 🎯 Project Overview

This project implements an **automated anomaly detection system** for machinery sounds using deep learning and transfer learning techniques. The system classifies audio signals from industrial machinery as either **Normal** or **Abnormal** by converting audio data into mel spectrograms and leveraging pretrained ResNet18 models.

**Key Achievement:** 94% validation accuracy using optimized transfer learning approach


## 🏆 Problem Statement

Industrial machinery failures can lead to costly downtime and safety hazards. This project addresses the challenge of **automatic fault detection in machinery** by analyzing acoustic signatures to identify abnormal operating conditions before catastrophic failures occur.


## 🧠 Technical Approach

### Core Methodology
- **Input:** Audio files (WAV format) from machinery
- **Preprocessing:** Convert audio → Mel Spectrograms (visual representation)
- **Model:** ResNet18 (pretrained on ImageNet)
- **Transfer Learning:** Fine-tune CNN for audio classification
- **Output:** Binary classification (Normal/Abnormal)

### Why Mel Spectrograms?
Mel spectrograms provide a time-frequency representation of audio that captures the essential characteristics of machinery sounds. By converting 1D audio signals into 2D images, we can leverage powerful computer vision models like ResNet18 for classification.


## 📊 Three Model Variants - Experimental Comparison

We trained **three different configurations** of ResNet18 to find the optimal transfer learning strategy:

### 🥉 Model 1: Minimal Fine-tuning
**Strategy:** Train only the final classification layer
- **Frozen Layers:** All convolutional layers (layer1-4)
- **Trainable:** Final FC layer only
- **Optimizer:** SGD (lr=0.001, momentum=0.9)
- **Scheduler:** None
- **Epochs:** 20
- **Early Stopping:** No
- **Data Augmentation:** Basic (RandomResizedCrop, RandomHorizontalFlip)
- **Validation Accuracy:** **86%**

**Rationale:** Minimal parameter updates, fastest training, leverages pretrained features maximally.


### 🥈 Model 2: Moderate Fine-tuning
**Strategy:** Unfreeze the last residual block + FC layer
- **Frozen Layers:** layer1, layer2, layer3
- **Trainable:** layer4 (last residual block) + FC layer
- **Optimizer:** SGD (lr=0.0001, momentum=0.9)
- **Scheduler:** StepLR (step_size=5, gamma=0.1)
- **Epochs:** 25 (with early stopping)
- **Early Stopping:** Yes (patience=3)
- **Data Augmentation:** Basic (RandomResizedCrop, RandomHorizontalFlip)
- **Validation Accuracy:** **87%**

**Rationale:** Allows deeper layers to adapt to spectrogram features while keeping early feature extractors frozen.


### 🥇 Model 3: Aggressive Fine-tuning ⭐ BEST PERFORMING
**Strategy:** Unfreeze last two residual blocks + advanced augmentation
- **Frozen Layers:** layer1, layer2
- **Trainable:** layer3, layer4 + FC layer
- **Optimizer:** Adam (lr=0.00005)
- **Scheduler:** StepLR (step_size=5, gamma=0.1)
- **Epochs:** 25 (with early stopping)
- **Early Stopping:** Yes (patience=3)
- **Data Augmentation:** Advanced
  - RandomResizedCrop(224)
  - RandomHorizontalFlip()
  - RandomRotation(10°)
  - ColorJitter(brightness=0.1, contrast=0.1)
- **Validation Accuracy:** **94%** ✨

**Rationale:** More trainable parameters + Adam optimizer + enhanced augmentation allows the model to better adapt to the specific characteristics of machinery spectrograms.


## 📈 Results Comparison

| Model | Trainable Layers | Optimizer | Learning Rate | Augmentation | Val Accuracy |
|-------|------------------|-----------|---------------|--------------|--------------|
| Model 1 | FC only | SGD | 0.001 | Basic | **86%** |
| Model 2 | layer4 + FC | SGD | 0.0001 | Basic | **87%** |
| Model 3 🏆 | layer3 + layer4 + FC | Adam | 0.00005 | Advanced | **94%** |

**Key Findings:**
- 🔑 **Unfreezing more layers** (layer3 + layer4) provided better adaptation to spectrogram features
- 🚀 **Adam optimizer** outperformed SGD for this specific task
- 🎨 **Advanced data augmentation** (rotation + color jitter) improved generalization
- ⏱️ **Early stopping** prevented overfitting in Models 2 & 3
- 📉 **Learning rate scheduling** helped achieve better convergence


## 🗂️ Project Structure

```
├── model1.py                    # Minimal fine-tuning approach
├── model2.py                    # Moderate fine-tuning approach
├── model3.py                    # Best performing model (94% accuracy)
├── dataset/
│   ├── train/
│   │   ├── normal/             # Normal machinery spectrograms
│   │   └── abnormal/           # Abnormal machinery spectrograms
│   └── val/
│       ├── normal/
│       └── abnormal/
├── spectrogram_000001.zip      # Raw spectrogram data
└── README.md
```


## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install torch torchvision torchaudio
pip install librosa numpy matplotlib scikit-learn
```

**Required Libraries:**
- PyTorch >= 1.9.0
- torchvision
- NumPy
- Python 3.7+

### Dataset Preparation

1. **Organize your mel spectrogram images:**
```
content/
├── normal/          # PNG files of normal machinery sounds
└── abnormal/        # PNG files of abnormal machinery sounds
```

2. **The code automatically splits data (80/20):**
   - 80% for training
   - 20% for validation

### Training

**Train Model 1 (Quick baseline):**
```bash
python model1.py
```

**Train Model 2 (Moderate fine-tuning):**
```bash
python model2.py
```

**Train Model 3 (Best performance - Recommended):**
```bash
python model3.py
```

Models are automatically saved to Google Drive:
- `resnet18_final1.pth`
- `resnet18_final2.pth`
- `resnet18_final3.pth`


## 🔬 Implementation Details

### Image Preprocessing
All models use ImageNet normalization:
- **Input Size:** 224×224 pixels
- **Normalization:** Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
- **Batch Size:** 4

### Training Configuration
```python
# Common settings
Batch Size: 4
Device: CUDA (GPU) if available
Loss Function: CrossEntropyLoss
Classes: ['abnormal', 'normal']
```

### Data Augmentation Comparison

**Basic (Models 1 & 2):**
- RandomResizedCrop(224)
- RandomHorizontalFlip()

**Advanced (Model 3):**
- RandomResizedCrop(224)
- RandomHorizontalFlip()
- RandomRotation(10°)
- ColorJitter(brightness=0.1, contrast=0.1)


## 💡 Key Insights & Learnings

### 1. **Transfer Learning Works for Audio**
Despite being trained on natural images, ResNet18 successfully adapted to mel spectrograms, demonstrating the power of transfer learning across domains.

### 2. **Layer Unfreezing Strategy Matters**
- Freezing too many layers (Model 1) → Underfitting to domain-specific features
- Unfreezing deeper layers (Model 3) → Better adaptation to spectrogram characteristics
- Sweet spot: Unfreeze last 2 residual blocks

### 3. **Optimizer Choice Impact**
Adam's adaptive learning rates proved more effective than SGD for this task, likely due to the varying characteristics of different spectrogram features.

### 4. **Data Augmentation Boosts Performance**
Adding rotation and color jitter specifically helped the model:
- Handle variations in spectrogram intensity
- Become robust to temporal shifts in audio patterns
- Improve generalization to unseen machinery sounds

### 5. **Early Stopping is Essential**
Prevented overfitting and saved training time by stopping when validation loss plateaued.


## 🎯 Use Cases

- **Predictive Maintenance:** Detect early signs of machinery failure
- **Quality Control:** Monitor production line equipment health
- **Industrial IoT:** Real-time anomaly detection in smart factories
- **Cost Reduction:** Prevent unexpected downtime and repairs


## 🔮 Future Improvements

- [ ] **Ensemble Method:** Combine all three models for improved robustness
- [ ] **Multi-class Classification:** Detect specific fault types (bearing, gear, motor)
- [ ] **Real-time Inference:** Deploy on edge devices (Raspberry Pi, NVIDIA Jetson)
- [ ] **Explainability:** Add Grad-CAM visualizations to show what spectrograms features drive predictions
- [ ] **Alternative Architectures:** Test EfficientNet, Vision Transformers
- [ ] **Audio Augmentation:** Apply pitch shifting, time stretching directly on audio
- [ ] **Web Application:** Create Flask/FastAPI interface for easy deployment
- [ ] **Mobile App:** On-device inference for portable diagnostics



## 🛠️ Technologies Used

- **Deep Learning:** PyTorch, torchvision
- **Computer Vision:** ResNet18, Transfer Learning
- **Audio Processing:** Mel Spectrograms
- **Training:** Google Colab (GPU acceleration)
- **Optimization:** Adam, SGD, Learning Rate Scheduling, Early Stopping


## 👥 Team

**Developed by:** AISirens 
**Event:** TechFest - AI Paradox
**Date:** Oct 2024


## 🙏 Acknowledgments

- PyTorch team for excellent deep learning framework
- ImageNet pretrained models from torchvision
- Google Colab for free GPU resources
- [Hackathon Organizers] for hosting this challenge
- Open-source community for inspiration and tools


## 📄 License

MIT

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

*Built with ❤️ for predictive maintenance and industrial IoT*

</div>

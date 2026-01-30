# Machine Learning for Computer Vision

This repository contains three projects related to computer vision and machine learning applications.

---

## 📁 Project Structure

```
Machine-Learning-for-Computer-Vision/
├── CombineTexture_script/           # Blender-Unity texture converter
├── Image processing and computer vision/  # Rock-Paper-Scissors hand tracking game
└── Machine learning for vision and multimedia/  # Siamese ResNet50 for anime recognition
```

---

## 1. 🎨 CombineTexture Script

A Python script for converting PBR (Physically Based Rendering) textures between Blender and Unity formats.

### Description

This utility combines metalness and roughness texture maps into a single Unity-compatible metallic texture. Unity's Standard shader expects metalness in the Red channel and smoothness (inverse of roughness) in the Alpha channel of a single texture.

### Features

- Converts separate metalness and roughness maps into Unity's combined metallic texture format
- Automatically converts roughness to smoothness (inverts the texture)
- Supports both roughness and smoothness input maps (use `-s` flag for smoothness)
- Debug visualization mode with matplotlib

### Usage

```bash
python unityMetalTex.py <metalness_map> <roughness_map> [-s]
```

**Arguments:**
- `metalness_map`: Path to the metalness texture
- `roughness_map`: Path to the roughness texture (or smoothness with `-s` flag)
- `-s`: Optional flag if the second input is already a smoothness map instead of roughness

**Output:** `out.png` - Combined RGBA texture where:
- Red channel: Metalness
- Green channel: 0 (unused)
- Blue channel: 0 (unused)
- Alpha channel: Smoothness

### Dependencies

- OpenCV (`cv2`)
- Matplotlib (optional, for debug visualization)

---

## 2. ✊✋✌️ Rock-Paper-Scissors Hand Tracking Game

An interactive Rock-Paper-Scissors game using hand tracking and gesture recognition.

### Description

This project implements a real-time Rock-Paper-Scissors (Morra Cinese) game using computer vision and the MediaPipe library for hand tracking. The player competes against the computer by making hand gestures in front of a webcam.

### Features

- **Real-time hand tracking** using MediaPipe Hands solution
- **Gesture recognition** for rock (fist), paper (open hand), and scissors (two fingers)
- **State machine** with multiple game states: Menu, Tutorial, Game, and Result
- **Interactive UI** with button navigation using hand gestures
- **Hand orientation detection** (works with hand facing up or down)
- **Visual feedback** with countdown timer and result screens

### Game Flow

1. **Tutorial State**: Instructions on how to play
2. **Menu State**: Navigate to start game or view tutorial using hand pointer
3. **Game State**: 3-second countdown to make your gesture
4. **Result State**: Shows win/lose/draw result with the computer's choice

### Gesture Detection

The system tracks 21 hand landmarks and determines finger positions:
- **Rock (Sasso)**: All fingers closed (fist)
- **Paper (Carta)**: All fingers extended (open hand)
- **Scissors (Forbice)**: Only index and middle fingers extended

### Files

- `MorraCinese_TES.py`: Main game application with complete UI and game logic
- `MorraCinese_TES_6Secondi.py`: Alternative version with 6-second countdown
- `handTrackingModule_TES.py`: Hand detection module using MediaPipe
  - `handDetector` class with methods:
    - `findHands()`: Detects hands and draws landmarks
    - `findPosition()`: Returns landmark positions in pixel coordinates
    - `fingerUp()`: Returns list of open/closed finger states

### Dependencies

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- Matplotlib
- NumPy

### Running the Game

```bash
python MorraCinese_TES.py
```

**Note**: Requires a webcam connected to the computer.

---

## 3. 🎬 ShazamAnime - Siamese ResNet50 for Anime Recognition

A deep learning project using Siamese Neural Networks to recognize and classify anime from image frames.

### Description

This project addresses the challenge of few-shot learning for anime classification. It uses a Siamese Neural Network architecture with ResNet50 as the backbone to learn image similarity, enabling recognition of anime series from single frames extracted from trailers.

### Architecture

#### Siamese Network
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Feature extraction**: Global Average Pooling followed by a Dense layer (512 units)
- **L2 Normalization**: Applied to feature vectors for better similarity comparison
- **Distance metric**: Squared Euclidean distance between normalized feature vectors
- **Loss function**: Contrastive loss for learning similarity/dissimilarity

#### Alternative Architecture
- VGG-16 backbone option also implemented for comparison

### Training Strategy

1. **Warm-up Phase**: 
   - Freeze ResNet50 backbone layers
   - Train only the top Dense layers
   - Learning rate: Starting at 0.00001 with exponential decay

2. **Fine-tuning Phase**:
   - Gradually unfreeze backbone layers
   - Lower learning rate (0.000001)
   - Adaptive learning rate with callbacks

### Classification Methods

After training the Siamese network, two classification approaches are available:

1. **k-NN Classifier**:
   - Uses extracted feature embeddings
   - k=11 nearest neighbors
   - Euclidean distance metric
   - No additional training required

2. **Dense Layer Classifier**:
   - Additional trainable Dense layer on top of Siamese features
   - Sparse categorical cross-entropy loss
   - Requires additional training phase

### Data Visualization

- **t-SNE**: Embedding visualization using TensorBoard Projector
- **Sprite images**: Generated for visual exploration of the embedding space
- **Confusion matrices**: For model evaluation

### Dataset

The dataset is built from anime trailer videos:

#### Supported Anime Series
- A Silent Voice
- Attack on Titan
- Demon Slayer
- Jujutsu Kaisen
- Fullmetal Alchemist Brotherhood
- One Piece
- Death Note
- Dragon Ball
- Fruits Basket
- Gintama
- Hunter x Hunter
- Naruto
- My Hero Academia
- One Punch Man
- Re:ZERO
- Steins;Gate

### Dataset Creation Scripts

Located in `Dataset_creation/`:

1. **`data_set_extractor.py`**:
   - Extracts frames from trailer videos
   - Configurable frame sampling rate (default: every 10th frame)
   - Resizes frames to 512x512 pixels
   - Automatically splits into train/test sets (10% for test)
   - Organizes output by anime class labels
   - Exports metadata as JSON

2. **`make_siamese_dataset.py`**:
   - Creates pairs of images for Siamese training
   - Generates CSV with image paths and similarity labels
   - Label 0: Same anime (similar)
   - Label 1: Different anime (dissimilar)

3. **`make_siamese_dataset_balanced.py`**:
   - Creates balanced dataset for training
   - Ensures equal number of similar and dissimilar pairs
   - Random sampling for class balancing

### Core Notebooks

Located in `CoreCode/`:

1. **`ShazamAnime.ipynb`**:
   - Complete Siamese network implementation
   - Training pipeline (warm-up + fine-tuning)
   - Model surgery for feature extraction
   - Dense classifier training and evaluation

2. **`KNN-classifier.ipynb`**:
   - k-NN classification implementation
   - Feature extraction from trained Siamese model
   - Comparison with base ResNet50 (no Siamese training)
   - t-SNE visualization setup
   - Evaluation metrics and confusion matrices

### Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorBoard

### Note on Data

Due to storage constraints, the actual video files and extracted frames are not included in this repository. The dataset creation scripts are provided to allow reproduction of the dataset from locally obtained trailers.

---

## 📄 Documentation

Each project folder may contain additional documentation:
- **Image processing and computer vision**: `TESINA_Report.pdf` - Project report
- **Machine learning for vision and multimedia**: 
  - `Report_ShazamAnime_*.pdf` - Project reports (Italian and English versions)
  - `presentazione_ShazamAnime.pptx` - Project presentation
  - `Papers/` - Reference papers on Siamese networks and few-shot learning

---

## 👥 Authors

- Alessandro Bresciani
- Luca Filippetti (ShazamAnime project)
- Rovegno, Ciardo (Rock-Paper-Scissors project)

---

## 📜 License

Please refer to individual project folders for specific licensing information.

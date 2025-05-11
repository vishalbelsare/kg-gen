# Image Classification Project

## Project Overview
- Goal: Classify plant species from leaf images
- Dataset: 10,000 images across 50 species
- Model: [[Convolutional Neural Network]] architecture
- Metrics: Accuracy, Precision, Recall, F1-score

> This project demonstrates how computer vision can support biodiversity research

### Implementation Details
```python
# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='softmax')
])
```

## Performance Results
| Model | Accuracy | Training Time | Model Size |
|-------|----------|---------------|------------|
| Basic CNN | 78.4% | 2.3 hrs | 24MB |
| ResNet50 | 91.2% | 5.7 hrs | 98MB |
| EfficientNetB0 | 93.8% | 4.1 hrs | 29MB |

This project builds on concepts from [[note_07.md|Neural Networks Architecture]] and connects to the [[note_20.md|Transfer Learning Techniques]] I've been researching.

Next steps include implementing [[data augmentation]] to improve performance on underrepresented species.

#projects #machinelearning #computervision #deeplearning
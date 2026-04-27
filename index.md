---
layout: home
author: "Armando Sallas"
title: "Combining CNNs and Transformers for Smarter Galaxy Classification"
---

### How a hybrid model achieves high accuracy while reducing computation

---

## 🚀 Introduction

Galaxies come in different shapes—spiral, elliptical, and irregular—and these shapes help astronomers understand how the universe evolves.

However, modern telescopes generate massive amounts of image data, making manual classification impractical. This is where machine learning becomes essential.

---

## 🧠 The Challenge

Classifying galaxy images is not always straightforward:
- Some galaxies are easy to identify
- Others are ambiguous or noisy

Not every image requires the same level of analysis.

---

## 🔍 Existing Approaches

### CNNs (ResNet50)
- Fast and efficient
- Good at detecting local features
- May miss global structure

### Vision Transformers (ViT)
- Capture global relationships
- More powerful
- Computationally expensive

---

## 💡 Key Idea: A Hybrid System

Instead of choosing one model, I designed a system that uses both.

### How it works:
1. A CNN analyzes every image
2. If the prediction is confident → use it
3. If uncertain → pass to a Vision Transformer

This allows the system to focus computational power where it is needed most.

---

## ⚙️ Method

- CNN outputs class probabilities
- Confidence = highest probability
- If confidence < threshold → use ViT

This acts like a “second opinion” system.

---

## 🧪 Experimental Setup

- Dataset: Galaxy images (3 classes)
  - Elliptical
  - Spiral
  - Other
- Models:
  - CNN (ResNet50)
  - Vision Transformer (ViT)
  - Hybrid model
- Metrics:
  - Accuracy
  - Macro F1 Score

---

## 📊 Results

### Model Comparison

| Model | Accuracy | Macro F1 |
|------|--------|---------|
| CNN | 72.44% | 0.717 |
| ViT | 75.41% | 0.746 |
| Hybrid | **75.42%** | **0.746** |

👉 The hybrid model matches ViT performance.

---

## 📈 Tradeoff Analysis

As the confidence threshold increases:
- More images are sent to ViT
- Accuracy improves
- Computation increases

| Threshold | Accuracy | ViT Usage |
|----------|---------|----------|
| 0.60 | 74.36% | 25% |
| 0.80 | 75.38% | 63% |
| 0.95 | 75.42% | 86% |

This creates a tunable balance between speed and accuracy.

---

## 🧠 Key Insights

- CNN handles easy cases effectively
- ViT improves difficult cases
- Hybrid achieves the best of both:
  - High accuracy
  - Controlled computation

---

## ⚠️ Limitations

- Threshold must be manually tuned
- Hybrid system adds complexity

---

## 🔮 Future Work

- Learn the threshold automatically
- Apply to other computer vision tasks
- Optimize for real-time applications

---

## 🏁 Conclusion

This project demonstrates that combining models intelligently can outperform using a single model alone.

Instead of choosing between speed and accuracy, we can achieve both.

---

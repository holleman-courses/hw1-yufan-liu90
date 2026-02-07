# ECGR(5127) HW1 

## Student Information
**Name:** Yufan Liu  
**Student ID:** 801442680

---

## GitHub Repository
[https://github.com/holleman-courses/hw1-yufan-liu90](https://github.com/holleman-courses/hw1-yufan-liu90)

---
All calculations assume that each layer includes a bias term.  

---
# 1. Analysis Problems

## 1.1 Problem 1  
Fully-Connected Network (8-bit)

- Input size: 120  
- Hidden layers: two layers with 256 units each  
- Output classes: 10  
- Weights and activations are stored as 8-bit values (1 byte each)

### A. Parameter Storage

Layer 1 (120 → 256):  
- Weights: 120 × 256 = 30,720  
- Biases: 256  
- Total: 30,976  

Layer 2 (256 → 256):  
- Weights: 256 × 256 = 65,536  
- Biases: 256  
- Total: 65,792  

Layer 3 (256 → 10):  
- Weights: 256 × 10 = 2,560  
- Biases: 10  
- Total: 2,570  

Total number of parameters:  
30,976 + 65,792 + 2,570 = 99,338  

Since parameters are 8-bit (1 byte each):

**Answer:**  
99,338 bytes

---

### B. MACs per Inference

Each weight contributes one MAC operation.

- Layer 1: 120 × 256 = 30,720  
- Layer 2: 256 × 256 = 65,536  
- Layer 3: 256 × 10 = 2,560  

Total MACs:  
30,720 + 65,536 + 2,560 = 98,816  

**Answer:**  
98,816 MACs

---

### C. Temporary SRAM Requirement

Layer output sizes (in bytes):

- Input layer: 120  
- Hidden layer 1: 256  
- Hidden layer 2: 256  
- Output layer: 10  

Simultaneous storage for consecutive layers:

- Input + Hidden 1: 120 + 256 = 376  
- Hidden 1 + Hidden 2: 256 + 256 = 512  
- Hidden 2 + Output: 256 + 10 = 266  

The maximum requirement is 512 bytes.

**Answer:**  
512 bytes of SRAM

---

## 1.2 Problem 2  
Fully-Connected Network (32-bit)

- Input size: 1280  
- Hidden layers: two layers with 512 units each  
- Output classes: 32  
- Weights and activations are stored as 32-bit values (4 bytes each)

### A. Parameter Storage

Layer 1 (1280 → 512):  
- Weights: 1280 × 512 = 655,360  
- Biases: 512  
- Total: 655,872  

Layer 2 (512 → 512):  
- Weights: 512 × 512 = 262,144  
- Biases: 512  
- Total: 262,656  

Layer 3 (512 → 32):  
- Weights: 512 × 32 = 16,384  
- Biases: 32  
- Total: 16,416  

Total number of parameters:  
655,872 + 262,656 + 16,416 = 934,944  

Each parameter is 4 bytes.

Total parameter storage:  
934,944 × 4 = 3,739,776 bytes

**Answer:**  
3,739,776 bytes (approximately 3.74 MB)

---

### B. Inference Time

Total MACs required:  
1280 × 512 + 512 × 512 + 512 × 32  
= 655,360 + 262,144 + 16,384  
= 933,888 MACs

Processor speed:  
- Clock frequency: 80 MHz  
- One MAC every 4 cycles  

Effective throughput:  
80 MHz / 4 = 20 million MACs per second

Inference time:  
933,888 / 20,000,000 = 0.0467 seconds

**Answer:**  
46.7 ms

---

### C. Temporary SRAM Requirement

Layer output sizes (number of values):

- Input layer: 1280  
- Hidden layer 1: 512  
- Hidden layer 2: 512  
- Output layer: 32  

Maximum simultaneous storage:  
1280 + 512 = 1,792 values  

Each value is 4 bytes.

Total SRAM required:  
1,792 × 4 = 7,168 bytes

**Answer:**  
7,168 bytes of SRAM (approximately 7 KB)

---

## 1.3 Problem 3  
Convolutional Neural Network (8-bit)

- Input size: 80 × 80 × 3  
- Two convolutional layers, each with 128 channels  
- Kernel size: 3 × 3  
- Padding: "same"  
- Followed by a flatten layer and a 10-unit output layer  
- Weights and activations are stored as 8-bit integers (1 byte each)

### A. Parameter Storage

Convolution Layer 1 (3 input channels → 128 output channels):  
- Weights: 3 × 3 × 3 × 128 = 3,456  
- Biases: 128  
- Total: 3,584  

Convolution Layer 2 (128 → 128):  
- Weights: 3 × 3 × 128 × 128 = 147,456  
- Biases: 128  
- Total: 147,584  

Flattened size after second convolution:  
80 × 80 × 128 = 819,200  

Dense output layer (819,200 → 10):  
- Weights: 819,200 × 10 = 8,192,000  
- Biases: 10  
- Total: 8,192,010  

Total number of parameters:  
3,584 + 147,584 + 8,192,010 = 8,343,178  

**Answer:**  
8,343,178 bytes (approximately 8.34 MB)

---

### B. Inference Time

Processor speed:  
- Clock frequency: 120 MHz  
- 4 MACs per cycle  

Effective throughput:  
120 MHz × 4 = 480 million MACs per second

MACs required:

- Convolution Layer 1:  
  80 × 80 × 128 × (3 × 3 × 3) = 22,118,400 MACs  

- Convolution Layer 2:  
  80 × 80 × 128 × (3 × 3 × 128) = 943,718,400 MACs  

- Dense layer:  
  819,200 × 10 = 8,192,000 MACs  

Total MACs:  
22,118,400 + 943,718,400 + 8,192,000 = 974,028,800 MACs

Inference time:  
974,028,800 / 480,000,000 = 2.03 seconds

**Answer:**  
2,029 ms (approximately 2.03 seconds)

---

### C. Temporary SRAM Requirement

Layer output sizes (number of values):

- Input: 80 × 80 × 3 = 19,200  
- Convolution Layer 1 output: 80 × 80 × 128 = 819,200  
- Convolution Layer 2 output: 80 × 80 × 128 = 819,200  
- Output layer: 10  

Simultaneous storage for consecutive layers:

- Input + Conv1: 838,400  
- Conv1 + Conv2: 1,638,400  
- Conv2 + Output: 819,210  

The maximum requirement is 1,638,400 bytes.

**Answer:**  
1,638,400 bytes of SRAM (approximately 1.64 MB)


# 2. Training Results Analysis and Model Comparison

## 2.1 Sub-50k Parameter Model Performance

The final sub-50k parameter model contains **43,690 parameters**, satisfying the constraint of using fewer than 50,000 parameters. The model was trained for **30 epochs** on the CIFAR-10 dataset using an adaptive learning rate schedule (**ReduceLROnPlateau**).

At the end of training, the model achieved:

- **Final training accuracy:** approximately **87.3%**
- **Best validation accuracy:** **76.48%**
- **Final test accuracy:** **76.20%**

The validation accuracy steadily improved during the early stages of training and continued to increase more gradually after learning rate reductions. The best validation accuracy was achieved at the final epoch, indicating that the learning rate schedule helped the model converge to a better minimum.

---

## 2.2 Training Time

The average training time per epoch ranged from **22 to 26 seconds**, with some longer epochs occurring during learning rate transitions. The total training time for 30 epochs was approximately **12–13 minutes** on a CPU-only system(I am currently resolving the version compatibility issues between TensorFlow, CUDA, and cuDNN, and will switch to GPU training afterward.). This training time is reasonable given the depth of the convolutional architecture and the absence of GPU acceleration.

---

## 2.3 Overfitting Analysis

Overfitting was **moderate but well-controlled** in this model:

- Training accuracy continued to increase throughout training, reaching over **87%**.
- Validation accuracy plateaued around **76%**, with small fluctuations after epoch 20.
- The gap between training and validation accuracy remained around **10–11%**, which is expected for CIFAR-10 on a compact model.

The use of **Batch Normalization**, **learning rate decay**, and a limited parameter budget effectively mitigated severe overfitting. Although the model continued to slightly improve on the training set after epoch 20, validation performance remained stable, suggesting that training for significantly more epochs would likely yield diminishing returns.

---

## 2.4 Comparison with Other Models

Compared to the earlier models explored in this assignment:

- **Model 1 (Fully-Connected Network)** trained quickly but achieved significantly lower accuracy, indicating limited representational capacity for image data.
- **Model 2 (Standard CNN)** achieved higher accuracy than the fully-connected model but required substantially more parameters and longer training time.
- **Model 3 (Depthwise-Separable CNN)** provided a strong trade-off between parameter efficiency and accuracy.

The final sub-50k model outperformed the simpler architectures while remaining within the strict parameter budget. It achieved competitive accuracy with reasonable training time, demonstrating that architectural choices such as deeper convolutional stacks, batch normalization, and learning rate scheduling have a larger impact on performance than simply increasing parameter count.

---

## 2.5 Summary

Overall, the sub-50k parameter model achieved a strong balance between **accuracy**, **efficiency**, and **generalization**. The final test accuracy of **76.2%** exceeds the minimum requirement and shows that careful architectural and training design can yield high performance even under tight parameter constraints.

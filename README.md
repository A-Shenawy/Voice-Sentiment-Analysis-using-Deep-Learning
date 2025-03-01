# 🎤 Voice Sentiment Analysis Using Deep Learning  

<img src="https://github.com/A-Shenawy/Voice-Sentiment-Analysis/blob/main/VSA.PNG" width="100%" style="background-size: cover;">

---

## **📌 Project Overview**  
This project aims to classify **emotions from voice recordings** using **deep learning techniques**. Human speech carries rich emotional cues, and analyzing these can enhance **human-computer interaction, mental health monitoring, and customer experience analysis**. The model is trained on the **EYASE dataset**, which contains **Arabic voice samples** labeled with emotions: **happy, sad, angry, and neutral**.  

### **🎯 Key Objectives**  
✅ Develop a **CNN-based deep learning model** to classify emotions from speech.  
✅ Preprocess and **augment** voice data to improve model generalization.  
✅ Extract **Mel-Spectrogram** features for deep learning-based classification.  
✅ Evaluate model performance using **accuracy, F1-score, precision, and recall**.  

---

## **📊 Dataset Details**  
- **Total audio files**: 579  
- **Sentiment classes**:  
  - **Happy**: 132  
  - **Sad**: 147  
  - **Angry**: 150  
  - **Neutral**: 150  
- **Average sample rate**: **44.1 KHz**  
- **Average audio duration**: **2.33 sec**  
- **Data augmentation techniques applied**: **Noise addition, pitch shifting**  
- **Final dataset after augmentation**: **1,626 audio samples**  

---

## **🛠️ Methodology**  

### **1️⃣ Data Preprocessing & Augmentation**  
🔹 **Padding & truncating**: Standardizing audio length.  
🔹 **Data augmentation**:  
   - **Noise addition** → Simulates real-world background noise.  
   - **Pitch shifting** → Enhances model robustness to voice variations.  

### **2️⃣ Feature Extraction**  
🔹 **Mel-Spectrogram transformation**: Converts raw audio into spectrogram images.  
🔹 **Normalization & down-sampling**: Reducing sample rate for efficient processing.  

### **3️⃣ Model Architecture (CNN-Based Approach)**  
🔹 **Input**: Mel-Spectrogram images.  
🔹 **Convolutional Layers (Conv2D)** → Extracts spatial features.  
🔹 **MaxPooling Layers** → Reduces dimensionality.  
🔹 **Dropout Layers** → Prevents overfitting.  
🔹 **Flatten Layer** → Converts 2D features to 1D.  
🔹 **Dense Layers** → Classifies the emotions.  

### **4️⃣ Training & Validation**  
🔹 **Train-test split**: **80% training, 20% testing**.  
🔹 **Cross-validation**: **5-Fold K-Fold validation** to ensure robustness.  

---

## **📈 Results & Performance**  
📌 **Initial Model Accuracy**: **86.76%**  
📌 **Confusion Matrix Analysis**:  
   - **High precision & recall** for Angry and Sad classes.  
   - **Slight misclassification** in Neutral vs. Happy classes.  

📌 **K-Fold Cross-Validation Scores**:  
   - **Average Accuracy**: **87.7%**  
   - **Average F1 Score**: **87.63%**  
   - **Average Recall**: **87.7%**  
   - **Average Precision**: **87.92%**  

---

## **📌 Future Improvements**  
🔹 **Optimize model architecture** for better generalization.  
🔹 **Explore transformer-based models** (e.g., Wav2Vec2, Whisper) for speech processing.  
🔹 **Integrate emotion detection** with **real-time applications** (e.g., chatbots, virtual assistants).  

---

## **📦 Installation & Usage**  

### **🔧 Prerequisites**  
Make sure you have the following installed:  
- 🐍 Python 3.x  
- 🤖 TensorFlow / PyTorch  
- 🎵 Librosa (for audio processing)  
- 📊 NumPy, Pandas, Matplotlib  
- 📓 Jupyter Notebook  

### **🚀 Steps to Run the Project**  
1️⃣ **Clone the repository**  
   ```bash
   git clone https://github.com/A-Shenawy/Voice-Sentiment-Analysis.git
   cd Voice-Sentiment-Analysis

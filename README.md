
# 🌍 **Geoguessr AI**  
---

**Project Goal:**  
Develop an AI model to play Geoguessr, a geography-based game where players guess their location on Google Street View.

---

## 🗂️ **Project Structure**
---

The project is organized into the following directories:

- **`src`**: Contains source code, AI model, and data preprocessing scripts.
- **`notebooks`**: Jupyter notebooks for testing and training the AI model.
- **`data`**: Dataset for training the AI model.

---

## 💻 **Requirements**
---

- Python 3.8+
- PyTorch 1.9+
- Lightning 2.0+
- Hugging Face Transformers 4.10+

---

## ⚙️ **Installation**
---

Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

---

## 🚀 **Usage**
---

1. **Set .env file based on .env.template:**
 ```bash
HUGGINGFACE_TOKEN=""
KAGGLE_API_TOKEN=""
KAGGLE_USERNAME=""
 ```


2. **Download the necessary data**:

```bash
sh src/download_data.bash
```

3. **Train the AI model and upload to HuggingFace**:

```bash
python src/main.py
```

---

## 🧠 **Model Architecture**
---

The AI model is based on the **Tiny ViT** architecture, a lightweight version of the Vision Transformer (ViT). Key components include:

- **`tiny_vit_21m_224`**: A pre-trained Tiny ViT model with 21 million parameters and an input size of 224x224.
- **`TinyVitLightning`**: A custom LightningModule that wraps the Tiny ViT model with additional components, such as a classification head and a loss function.

---

## 📊 **Data**
---

The model is trained using two Kaggle datasets:

- [killusions/street-location-images](https://www.kaggle.com/datasets/killusions/street-location-images)
- [ubitquitin/geolocation-geoguessr-images-50k](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)

---

## 🙏 **Acknowledgments**
---

This project was inspired by the **Geoguessr** game and the **Tiny ViT** model. Special thanks to the creators of these amazing projects.
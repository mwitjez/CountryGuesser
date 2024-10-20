
# 🌍 **Country Guesser**  
A Vision Transformer (ViT)-based AI model for country classification from photos. The fine-tuned model achieves **85% accuracy** on the validation dataset.

---

## 🗂️ **Project Structure**  
The project is structured as follows:

- **`src/`**: Core source code, AI model, and data preprocessing scripts.
- **`notebooks/`**: Jupyter notebooks for model training and testing.
- **`data/`**: Datasets used for model training and validation.

---

## 💻 **Requirements**  
To run the project, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch 1.9+
- Lightning 2.0+
- Hugging Face Transformers 4.10+

---

## ⚙️ **Installation**  

1. **Install Dependencies**  
   To install the required Python packages, run:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, execute the following script for complete package installation:

   ```bash
   bash install_packages.sh
   ```

2. **Set Environment Variables**  
   Copy the template `.env.template` to `.env`, then fill in the required credentials:

   ```bash
   HUGGINGFACE_TOKEN=""
   KAGGLE_API_TOKEN=""
   KAGGLE_USERNAME=""
   ```

3. **Download Datasets**  
   Use the provided script to download the necessary data:

   ```bash
   bash download_data.bash
   ```

4. **Train the Model**  
   To train the model and upload it to Hugging Face, run:

   ```bash
   python src/main.py
   ```

---

## 🧠 **Model Architecture**  

The AI model utilizes the **Tiny ViT** architecture, designed for efficiency and accuracy. Key components include:

- **`tiny_vit_21m_224`**: A lightweight Vision Transformer pre-trained with 21 million parameters, optimized for 224x224 image inputs.
- **`TinyVitLightning`**: A custom PyTorch Lightning module that integrates the Tiny ViT model with additional components like a classification head and a loss function.

---

## 📊 **Datasets**  
The model is trained using the following datasets from Kaggle:

- [Street Location Images](https://www.kaggle.com/datasets/killusions/street-location-images)
- [Geolocation GeoGuessr Images 50k](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
- [Streetview Photospheres](https://www.kaggle.com/datasets/nikitricky/streetview-photospheres/data)

---

## 📄 **License**

This project is licensed under the MIT License, permitting free use, modification, and distribution. The software is provided "as-is," without any warranty. For details, see the [LICENSE](./LICENSE) file.

---
## 🙏 **Acknowledgments**  
This project draws inspiration from the popular game **GeoGuessr** and the advanced **Tiny ViT** architecture. Special thanks to the authors of these works for their valuable contributions to the community.
 - [TinyVit](https://github.com/wkcn/TinyViT)
 - [GeoGuessr](https://www.geoguessr.com)

**Geoguessr AI**
================

**Overview**
------------

This project aims to develop an AI model that can play the popular online game Geoguessr. Geoguessr is a geography-based game where players are dropped into a random location on Google Street View and must guess where they are in the world.

**Project Structure**
---------------------

This project is organized into the following directories:

* `src`: contains the source code for the project, including the AI model and data preprocessing scripts.
* `notebooks`: contains Jupyter notebooks for testing and training the AI model.
* `data`: contains the dataset used to train the AI model.

**Requirements**
---------------

* Python 3.8+
* PyTorch 1.9+
* Lightning 2.0+
* Hugging Face Transformers 4.10+

**Installation**
---------------

To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```
**Usage**
-----

To download necessary data run:
```bash
sh src/download_data.bash
```
To train the AI model and upload it to HuggingFace, run the following command:
```bash
python src/main.py
```
**Model Architecture**
----------------------

The AI model is based on a Tiny ViT architecture, which is a lightweight version of the Vision Transformer (ViT) model. The model consists of the following components:

* `tiny_vit_21m_224`: a pre-trained Tiny ViT model with 21 million parameters and a input size of 224x224.
* `TinyVitLightning`: a custom LightningModule that wraps the pre-trained Tiny ViT model and adds additional components, such as a classification head and a loss function.

**Data**
----------------------

Model is trained using two datsets from kaggle:
* killusions/street-location-images https://www.kaggle.com/datasets/killusions/street-location-images/
* ubitquitin/geolocation-geoguessr-images-50k https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k

**Acknowledgments**
----------------

This project was inspired by the Geoguessr game and the Tiny ViT model.
I would like to thank the creators of these projects for their work.
import os
import shutil

import lightning as L
import torch
from dotenv import load_dotenv
from huggingface_hub import Repository
from huggingface_hub import login

def upload_model(model: L.LightningModule) -> None:
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))

    model_save_path = "data/models/"
    repo = Repository(local_dir=model_save_path, clone_from="mwitjez/geoguessr_tiny_ViT")

    input_sample = torch.randn(1, 3, 224, 224)
    model.to_onnx(f"{model_save_path}/model.onxx", input_sample, export_params=True)
    torch.save(model.state_dict(), f"{model_save_path}/geoguessr_model.bin")

    repo.push_to_hub()

    if os.path.exists(model_save_path):
        shutil.rmtree(model_save_path)
        print(f"'{model_save_path}' has been removed successfully.")
    else:
        print(f"'{model_save_path}' does not exist.")

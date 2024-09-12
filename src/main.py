from train import train
from upload_model import upload_model


if __name__ == "__main__":
    trained_model = train()
    upload_model(trained_model)

import json
import argparse

from train import train
from upload_model import upload_model


def get_args():
    parser = argparse.ArgumentParser(description="Training configuration options")

    parser.add_argument("--num_classes", type=int, help="Number of classes for the model")
    parser.add_argument("--pretrained", type=bool, help="Whether to use a pretrained model")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--image_size", type=int, help="Size of the input images")
    parser.add_argument("--trial_data", type=bool, help="Whether to use trial data")
    parser.add_argument("--no-upload_model", action="store_false", dest="upload_model", help="Skip uploading the trained model")

    return vars(parser.parse_args())


if __name__ == "__main__":
    with open("src/config/model_config.json", "r") as f:
        config = json.load(f)

    args = {k: v for k, v in get_args().items() if v is not None}
    config.update(args)

    trained_model = train(config)
    if args.get("upload_model"):
        upload_model(trained_model)

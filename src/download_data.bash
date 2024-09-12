#!/bin/bash

source .env

# List of datasets to download
datasets=(
    #"killusions/street-location-images"
    "ubitquitin/geolocation-geoguessr-images-50k"
)

download_dir="data/full_data"

# Set up your Kaggle API token (replace with your actual token)
#kaggle config set -n API_TOKEN -v "$KAGGLE_API_TOKEN"

# Loop through each dataset and download it
for dataset in "${datasets[@]}"; do
    # Extract the dataset name without the username (after the '/')
    dataset_name=$(basename "$dataset")

    # Download the dataset
    #kaggle datasets download -d "$dataset" -p "$download_dir"

    # Unzip the dataset if it is a zip file
    zip_file="$download_dir/${dataset_name}.zip"
    echo $zip_file
    if [[ -f "$zip_file" ]]; then
        unzip "$zip_file" -d "$download_dir"
        rm "$zip_file"
    fi
done
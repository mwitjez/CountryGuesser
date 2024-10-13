#!/bin/bash

source .env

datasets=(
    "killusions/street-location-images"
    "ubitquitin/geolocation-geoguessr-images-50k"
    "nikitricky/streetview-photospheres"
)

download_dir="data/full_data"

kaggle config set -n API_TOKEN -v "$KAGGLE_API_TOKEN"

for dataset in "${datasets[@]}"; do
    dataset_name=$(basename "$dataset")

    kaggle datasets download -d "$dataset" -p "$download_dir"

    zip_file="$download_dir/${dataset_name}.zip"
    echo $zip_file
    if [[ -f "$zip_file" ]]; then
        unzip "$zip_file" -d "$download_dir"
        rm "$zip_file"
    fi
done
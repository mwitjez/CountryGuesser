import json
import os
from io import BytesIO

import msgpack
import pandas as pd
import reverse_geocoder as rg
from PIL import Image
from tqdm import tqdm


class LargeDatasetPreprocessor:
    def __init__(self):
        pass

    def _get_image(self, record):
        return Image.open(BytesIO(record["image"]))

    def _images_read_save(self, shard_fnames, base_path):
        directory = "large-dataset-of-geotagged-images"
        path = os.path.join(base_path, directory)
        if not os.path.exists(path):
            os.mkdir(path)
            print("Directory '% s' created" % directory)
        coords = []
        image_id = []
        for shard_fname in tqdm(shard_fnames):
            with open(shard_fname, "rb") as infile:
                unpacker = msgpack.Unpacker(infile, raw=False)
                for record in unpacker:
                    image = self._get_image(record)
                    image_path = os.path.join(
                        base_path,
                        directory,
                        f"{record['latitude']}_{record['longitude']}_{record['id'].replace('/', '')}",
                    )
                    image.save(image_path, "JPEG")
                    coords.append((record["latitude"], record["longitude"]))
                    image_id.append(image_path)

        return coords, image_id

    def _create_df(self, image_id, coords):
        df_1 = pd.DataFrame(image_id)
        df_1.rename(columns={df_1.columns[0]: "id"}, inplace=True)
        df_2 = pd.DataFrame(coords)
        df_2.rename(
            columns={df_2.columns[0]: "lat", df_2.columns[1]: "lng"}, inplace=True
        )
        df_final = pd.concat([df_1, df_2], axis=1)
        return df_final

    def get_data(self, base_path):
        dataset_dir = f"{base_path}/shards/"
        shard_fnames = [
            os.path.join(dataset_dir, fname)
            for fname in os.listdir(dataset_dir)
            if fname.endswith(".msg")
        ]
        coords, image_id = self._images_read_save(shard_fnames[:10], base_path)

        df = self._create_df(image_id, coords)

        coordinates = list(zip(df["lat"], df["lng"]))
        results = rg.RGeocoder(mode=2).query(coordinates)
        country_code_to_label = json.load(
            open(f"{base_path}/country_code_to_index.json", "r")
        )
        df["label"] = [country_code_to_label.get(result["cc"], pd.NA) for result in results]

        data = [(row_id, int(label)) for row_id, label in zip(df["id"], df["label"]) if pd.notna(label)]
        return data

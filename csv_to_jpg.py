import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def csv_to_images(csv_path, output_dir, img_size=28):
    """
    Convert a CSV file containing image data to PNG images."""
    # Load CSV
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
        label = int(row[0])
        pixels = np.array(row[1:], dtype=np.uint8).reshape((img_size, img_size))

        # Save image as label_index.png (e.g., 3_150.png)
        img = Image.fromarray(pixels, mode="L")
        filename = f"{label}_{index}.png"
        img.save(os.path.join(output_dir, filename))

    print(f"Saved {len(df)} images to '{output_dir}'")


if __name__ == "__main__":
    csv_to_images("data/sign_mnist_train_v1.csv", "data/sign_mnist_train")
    csv_to_images("data/sign_mnist_test_v1.csv", "data/sign_mnist_test")

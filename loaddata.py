import os
import kagglehub
import numpy as np
import cv2
from tqdm import tqdm

# =========================
# STEP 1: DOWNLOAD DATASET
# =========================
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("andrewmvd/liver-tumor-segmentation")

print("Downloaded to:", dataset_path)

# =========================
# STEP 2: OUTPUT STRUCTURE
# =========================
OUTPUT_PATH = "processed_dataset"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return (img * 255).astype(np.uint8)

# =========================
# CONVERSION FUNCTION
# =========================
def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, file)

        try:
            if file.endswith(".npy"):
                img = np.load(file_path)
                img = normalize(img)

                save_path = os.path.join(output_folder, file.replace(".npy", ".png"))
                cv2.imwrite(save_path, img)

            elif file.endswith(".png") or file.endswith(".jpg"):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = normalize(img)

                save_path = os.path.join(output_folder, file)
                cv2.imwrite(save_path, img)

        except Exception as e:
            print(f"Skipping {file}: {e}")

# =========================
# STEP 3: AUTO-DETECT STRUCTURE
# =========================
print("Processing dataset...")

for root, dirs, files in os.walk(dataset_path):
    for d in dirs:
        folder_path = os.path.join(root, d)

        if "image" in d.lower():
            out = os.path.join(OUTPUT_PATH, "images")
            process_folder(folder_path, out)

        elif "mask" in d.lower():
            out = os.path.join(OUTPUT_PATH, "masks")
            process_folder(folder_path, out)

print("✅ DONE: Dataset ready in 'processed_dataset'")
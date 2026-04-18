import os
import numpy as np
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models

def get_paths(img_dir, mask_dir):
    img_files = os.listdir(img_dir)

    img_paths = []
    mask_paths = []

    for img_name in img_files:
        base = img_name.replace(".tif", "").replace(".png", "")
        mask_name = base + "_mask.tif"

        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            img_paths.append(img_path)
            mask_paths.append(mask_path)

    return img_paths, mask_paths
def generator(image_paths, mask_paths, batch_size=8):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_imgs = []
            batch_masks = []

            for img_path, mask_path in zip(
                image_paths[i:i+batch_size],
                mask_paths[i:i+batch_size]
            ):
                img = cv2.imread(img_path, 0)
                img = cv2.resize(img, (128, 128)) / 255.0

                mask = cv2.imread(mask_path, 0)
                mask = cv2.resize(mask, (128, 128)) / 255.0

                batch_imgs.append(img)
                batch_masks.append(mask)

            yield (
                np.array(batch_imgs)[..., np.newaxis],
                np.array(batch_masks)[..., np.newaxis]
            )
# =========================================================
# 🔥 DATASET SETUP (ONE-TIME PIPELINE)
# =========================================================

def setup_dataset():
    import kagglehub

    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    IMG_DIR = os.path.join(PROCESSED_DIR, "images")
    MASK_DIR = os.path.join(PROCESSED_DIR, "masks")

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # Skip if already processed
    if len(os.listdir(IMG_DIR)) > 50:
        print("✅ Dataset already ready")
        return IMG_DIR, MASK_DIR

    print("🚀 Downloading dataset...")
    dataset_path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

    def normalize(img):
        img = img.astype("float32")
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
        return (img * 255).astype("uint8")

    def process(file_path):
        name = os.path.basename(file_path).lower()
        try:
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                img = normalize(img)

                if "_mask" in name:
                    cv2.imwrite(os.path.join(MASK_DIR, name), img)
                else:
                    cv2.imwrite(os.path.join(IMG_DIR, name), img)
        except Exception as e:
            print("Error processing:", file_path, e)

    for root, _, files in os.walk(dataset_path):
        for f in tqdm(files):
            process(os.path.join(root, f))

    print("✅ Dataset processed")
    return IMG_DIR, MASK_DIR


# =========================================================
# 🔥 DATA LOADER (FAST + ALIGNED)
# =========================================================

"""def load_data(img_dir, mask_dir):
    images, masks = [], []

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(img_files, mask_files):
        img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        img = img / 255.0
        mask = mask / 255.0

        images.append(img)
        masks.append(mask)

    X = X = np.array(images, dtype=np.float32).reshape(-1, 256, 256, 1)
    y = X = np.array(masks, dtype=np.float32).reshape(-1, 256, 256, 1)
    print("Dataset size:", len(X))
    return X, y"""


# =========================================================
# 🔥 OPTIMIZED SEGNET MODEL
# =========================================================

def build_segnet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================================
# 🔥 TRAINING (FAST + CONTROLLED)
# =========================================================

def train():
    workers=2,
    use_multiprocessing=True
    IMG_DIR, MASK_DIR = setup_dataset()

    print("📂 Preparing paths...")
    img_paths, mask_paths = get_paths(IMG_DIR, MASK_DIR)
    img_paths = img_paths[:800]
    mask_paths = mask_paths[:800]

    print("🧠 Building model...")
    model = build_segnet()

    print("⚡ Training...")

    batch_size = 8

    model.fit(
        generator(img_paths, mask_paths, batch_size),
        steps_per_epoch=len(img_paths) // batch_size,
        epochs=10
    )

    model.save("segnet_model.keras")
    print("✅ Model saved")


# =========================================================
# 🔥 INFERENCE (FAST)
# =========================================================

def predict(image_path, model_path="segnet_model.h5"):
    model = tf.keras.models.load_model(model_path)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0

    img = img.reshape(1, 128,128, 1)

    pred = model.predict(img)[0]
    pred = (pred > 0.5).astype("uint8") * 255

    out_path = "output.png"
    cv2.imwrite(out_path, pred)

    return out_path


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    train()
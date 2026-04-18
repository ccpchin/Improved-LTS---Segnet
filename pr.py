import numpy as np
import os
import tensorflow as tf
import pydicom
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load DICOM Images and Masks
class LiverTumorDataset:
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
    
    def load_dicom(self, path):
        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize
        return image
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = self.load_dicom(img_path)
        mask = self.load_dicom(mask_path) > 0  # Convert to binary mask
        
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        mask = np.expand_dims(mask, axis=-1)
        
        return image, mask
    
    def __len__(self):
        return len(self.images)

# Create Dataset
image_dir = "C:/Users/chinm/OneDrive/Desktop/3Dircadb1.1/3Dircadb1.1/PATIENT_DICOM/PATIENT_DICOM"
mask_dir = "C:/Users/chinm/OneDrive/Desktop/3Dircadb1.1/3Dircadb1.1/LABELLED_DICOM/LABELLED_DICOM"
dataset = LiverTumorDataset(image_dir, mask_dir)

X, Y = [], []
for i in range(len(dataset)):
    image, mask = dataset[i]
    X.append(image)
    Y.append(mask)

X = np.array(X)
Y = np.array(Y)

# Define SegNet Model
def build_segnet(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Decoder
    up1 = UpSampling2D((2, 2))(pool3)
    deconv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(up1)
    
    up2 = UpSampling2D((2, 2))(deconv1)
    deconv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    
    up3 = UpSampling2D((2, 2))(deconv2)
    deconv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(deconv3)
    
    model = Model(inputs, outputs)
    return model

# Initialize and Compile Model
input_shape = (X.shape[1], X.shape[2], 1)
model = build_segnet(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, Y, batch_size=4, epochs=10, validation_split=0.2)

# Save Model
model.save("segnet_liver_tumor.keras")

# Visualization
idx = 0  # Change index to visualize different samples
sample_image, sample_mask = X[idx], Y[idx]
prediction = model.predict(sample_image[np.newaxis, ...])[0]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(sample_image.squeeze(), cmap='gray')
ax[0].set_title("CT Scan")
ax[1].imshow(sample_mask.squeeze(), cmap='jet')
ax[1].set_title("Ground Truth")
ax[2].imshow(prediction.squeeze(), cmap='jet')
ax[2].set_title("Prediction")
plt.show()
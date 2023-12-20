import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
from numpy import resize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16  # Example: VGG16 model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Cataract, Glaucoma are of 256x256
# Diabetic_retinopathy, Normal are of 512x512

# # Defining the image dimensions
image_height = 256
image_width = 256
#
# # Load the pre-trained CNN model:
#
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
print('VGG16 model loaded')
#
# # Creating a new model for feature extraction
#
model = Model(inputs=base_model.input, outputs=base_model.output)
print('model initiated')


#
# # Extracting Features from the dataset
#
# # Define your dataset directory
# dataset_directory = 'eye_diseases'
#
#
# # Load images and preprocess them
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(image_height, image_width, 3))
    imgee_array = image.img_to_array(img)
    imgee_array = preprocess_input(imgee_array)
    return imgee_array


#
# # Extract features for each image in the dataset
# def extract_features_from_dataset(datasetdirectory):
#     features = []
#
#     for class_folder in os.listdir(datasetdirectory):
#         class_path = os.path.join(datasetdirectory, class_folder)
#         print('Processing class folder: {}'.format(class_path))
#         for img_name in os.listdir(class_path):
#             img_path = os.path.join(class_path, img_name)
#             img_array = load_and_preprocess_image(img_path)
#             img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#             features.append(model.predict(img_array).flatten())  # Extract features using the model
#     print('Features loaded')
#
#     return np.array(features)
#
#
# # Extract features from the dataset
# extracted_features = extract_features_from_dataset('eye_diseases')
# print(extracted_features)
#
# with open('extracted_features1.pkl', 'wb') as file:
#     pickle.dump(extracted_features, file)
#
#
# # Applying Kmeans clustering

extracted_features = np.load('extracted_features2.npy')
#
# n_clusters = 4  # Number of clusters ('cataract', 'diabetic_retinopathy', 'glaucoma', 'normal')
# kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

# Fitting the features
# print('Fitting features...')
# kmeans.fit(extracted_features)
# print('Saving model...')
# joblib.dump(kmeans, 'kmeans.pkl')
# print('Kmeans model saved.')
# Apply PCA to reduce dimensionality to 2 components for visualization

# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(extracted_features)


# Saving reduced features for future use
# np.save('reduced_features.npy', reduced_features)

intermediate_layer = 'block5_conv3'
upsample_model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer).output)

kmeans = joblib.load('kmeans.pkl')

print(kmeans.cluster_centers_.shape)
print(kmeans.inertia_)

original_image = image.load_img("eye_diseases\glaucoma\_0_4517448.jpg", target_size=(image_height, image_width, 3))
original_image = image.img_to_array(original_image)

img_array = load_and_preprocess_image("eye_diseases\glaucoma\_0_4517448.jpg")
img_array = np.expand_dims(img_array, axis=0)
print(img_array.shape)
img_feature = model.predict(img_array).flatten()
x = kmeans.predict(img_feature.reshape(1, -1))
print(x)
print(kmeans.cluster_centers_)

scaler = MinMaxScaler(feature_range=(0, 255))

upsampled_features = scaler.fit_transform(kmeans.cluster_centers_[x])
print(upsampled_features.shape)
upsampled_features = resize(upsampled_features, (1, 256, 256, 3))
upsampled_features = upsample_model.predict(upsampled_features)
upsampled_features = resize(upsampled_features, (1, 256, 256, 3))

plt.subplot(1, 2, 1)
plt.imshow(original_image.astype(np.uint8))
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(upsampled_features[0].astype(np.uint8))
plt.title("Reconstructed Image")

plt.show()

# print("\nprinting extracted features...\n", extracted_features.shape, "\n", extracted_features,
#       "\n\nprinting reduced features...\n", reduced_features.shape, "\n", reduced_features)

# Reshape the data to be a list of 64-dimensional vectors (8x8 images)

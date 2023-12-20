import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os
import numpy as np
from skimage import io, color, feature, exposure, transform
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# Function to extract HOG features from an image and visualize it
def extract_hog_features(image):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Calculate HOG features
    hog_features, hog_image = feature.hog(gray_image, visualize=True)

    # Enhance the contrast of the HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_features, hog_image_rescaled


#
# # Path to the root folder of your dataset
# dataset_path = 'eye_diseases'
#
# # List all subdirectories (assuming each subdirectory corresponds to a class)
# class_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
# features_list = []
#
# # Loop through each class folder
# for class_folder in class_folders:
#     print("Processing class...", class_folder, "\n\n\n\n\n")
#     class_name = os.path.basename(class_folder)
#     # Loop through each image in the class folder
#     for image_filename in os.listdir(class_folder):
#         image_path = os.path.join(class_folder, image_filename)
#
#         # Cataract, Glaucoma are of 256x256
#         # Diabetic_retinopathy, Normal are of 512x512
#
#         # Load the image
#         image = io.imread(image_path)
#         image = transform.resize(image, (256, 256, 3))
#         # Extract HOG features and visualize
#         hog_features, hog_image = extract_hog_features(image)
#
#         # Append HOG features to the features list
#         print("appending features into the list...")
#         features_list.append(hog_features)
#
# # Convert lists to NumPy arrays
# features_array = np.array(features_list)
# print(features_array)
# np.random.shuffle(features_array)
# np.save('hog_features.npy', features_array)

features_array = np.load("hog_features.npy")

org_image = "eye_diseases\glaucoma\_100_965860.jpg"
org_image = io.imread(org_image)
org_image = transform.resize(org_image, (256, 256, 3))

feat, hog_org_image = extract_hog_features(org_image)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 2), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(org_image, cmap="gray")

ax2.axis('off')
ax2.imshow(hog_org_image, cmap="gray")

plt.show()

n_clusters = 4
# Number of clusters (0 through 9 digits)
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(features_array)
# joblib.dump(kmeans, 'kmeans_fitted_hog.pkl')
kmeans = joblib.load('kmeans_fitted_hog.pkl')

print(f'inertia: {kmeans.inertia_}')
# Visualize the cluster centers (representative images)
fig, ax = plt.subplots(1, n_clusters, figsize=(15, 3))

for i in range(n_clusters):
    center_image = kmeans.cluster_centers_[i].reshape(270, 270)  # Reshape to original dimensions
    ax[i].imshow(center_image, cmap='gray')
    ax[i].axis('off')
    ax[i].set_title(f'Cluster {i}')

plt.show()

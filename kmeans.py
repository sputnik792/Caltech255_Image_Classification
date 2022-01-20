# Preprocessing of images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# VGG16 and Models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# K-Means and PCA import
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Others
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle


# Set path to image dataset path
path = r"/Users/zunaidsorathiya/Documents/ALDAProject/Input Images"
os.chdir(path)

inputImages = []


# Adding all image names in the input List
with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.jpg') or file.name.endswith('.png'):
            inputImages.append(file.name)

# Pretrained VGG16 Model initialization
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    # reshape the image data to fit in the model
    reshaped_img = img.reshape(1, 224, 224, 3)
    # preprocess reshaped to fit in with Keras model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def perform_transformation(images, type):
    data = {}
    pkl_path = r"/Users/zunaidsorathiya/Documents/ALDAProject/"+type+"_image_features.pkl"
    # Extracting feature from the training set
    for image in images:
        # try to extract the features and update the dictionary
        try:
            feature = extract_features(image, model)
            data[image] = feature
        # Exception catch in the pickle file
        except:
            with open(pkl_path, 'wb') as file:
                pickle.dump(data, file)

    filenames = np.array(list(data.keys()))

    # Storing feature for all images in the feature vector
    feat = np.array(list(data.values()))

    # reshape to return samples of 4096 vectors
    feat = feat.reshape(-1, 4096)

    return feat, filenames

np.random.shuffle(inputImages)
split = int(len(inputImages)* 0.80)

# Splitting data into 80-20 train/test split
trainImages = inputImages[:split]
testImages = inputImages[split:]

# Setting number of objects to be clustered
unique_labels = 20
# Fetching all input images
path = r"/Users/zunaidsorathiya/Documents/ALDAProject/Input Images"
os.chdir(path)

# Preprocess train data
train_feat, train_filenames = perform_transformation(trainImages, "train")

# reducing number of features using PCA
pca_train = PCA(n_components=100)
pca_train.fit(train_feat)
x_train = pca_train.transform(train_feat)

# Fit the training data to form cluster 
kmeans = KMeans(n_clusters=unique_labels)
kmeans.fit(x_train)

# Clustering based on labels
groups = {}
for file, cluster in zip(train_filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

path = r"/Users/zunaidsorathiya/Documents/ALDAProject/Input Images/"

# Function to view all 5 samples of a particular cluster
def view_cluster(cluster):
    plt.figure(figsize=(10, 10));
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.suptitle('Cluster '+str(cluster+1),x=0.3,fontsize=15)
    files = groups[cluster]
    if len(files) > 5:
        files = files[:5]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(path + file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# Calling view_cluster to display 5 samples from each cluster
for cluster in range(len(groups)):
  view_cluster(cluster)


path = r"/Users/zunaidsorathiya/Documents/ALDAProject/Input Images/"
os.chdir(path)

# Preprocessing train data
test_feat, test_filenames = perform_transformation(testImages, "test")

# PCA on train data
x_test = pca_train.transform(test_feat)

# Test data class prediction
test_labels = kmeans.predict(x_test)


# Visualization test data
plt.figure(figsize=(50, 50))
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle('Test Data Set label prediction', fontsize=40)
for i in range(int(len(test_labels)/8)):
  plt.subplot(10, 10, i + 1)
  img = load_img(path + str(test_filenames[i]))
  img = np.array(img)
  plt.imshow(img)
  plt.axis('off')
  plt.title("Predicted Class: "+str(test_labels[i]+1),fontdict={'fontsize': 25})

# Unsuccessful attempt to integrate OpenCV based live Object Classification 
import cv2

cv2.startWindowThread()
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        print("Space button pressed")
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break   
cam.release()
cv2.destroyAllWindows() 

cam.release()
cv2.destroyAllWindows() 

path = r"/Users/zunaidsorathiya/Documents/ALDAProject"
os.chdir(path)
input_filename = ["opencv_frame_0.png"]
inp_feat, inp_filenames = perform_transformation(input_filename, "inputtest")

pca_train.fit(inp_feat)

x_inp = pca_inp.transform(inp_feat)

test_labels = kmeans.predict(x_inp)
print(test_labels[0])

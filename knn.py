import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Navigate to directory with folders for all categories

# Resize image
def resize_img(img_path, size):
    img = cv2.imread(img_path)
    return cv2.resize(img, size).flatten()

# Implement KNN classifier for 1 trial
def test_knn(categories, size=(32,32), test_size=0.25):
    # separate image paths into training and test sets
    # split within each category using stratified sampling
    train_paths = []
    test_paths = []
    for category in categories:
        image_paths = [os.path.join(category, img_name) for img_name in os.listdir(category)]
        tr_paths, te_paths = train_test_split(image_paths, test_size=test_size)
        train_paths.extend(tr_paths)
        test_paths.extend(te_paths)

    # generate labels for training and test sets
    train_labels = np.array([[categories.index(img_path.split('\\')[0])] for img_path in train_paths])
    test_labels = np.array([[categories.index(img_path.split('\\')[0])] for img_path in test_paths])

    # read images and resize to uniform size
    train_data = np.array([resize_img(img_path, size) for img_path in train_paths]).astype(np.float32)
    test_data = np.array([resize_img(img_path, size) for img_path in test_paths]).astype(np.float32)

    # create and train classifier
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # run for different numbers of neighbors
    knn1 = knn.findNearest(test_data, 1)
    knn3 = knn.findNearest(test_data, 3)
    knn5 = knn.findNearest(test_data, 5)
    knn9 = knn.findNearest(test_data, 9)

    # calculate accuracy
    acc_1nn = round(sum(knn1[1] == test_labels)[0] / len(test_labels), 6)
    acc_3nn = round(sum(knn3[1] == test_labels)[0] / len(test_labels), 6)
    acc_5nn = round(sum(knn5[1] == test_labels)[0] / len(test_labels), 6)
    acc_9nn = round(sum(knn9[1] == test_labels)[0] / len(test_labels), 6)
    
    return acc_1nn, acc_3nn, acc_5nn, acc_9nn

# Run multiple trials of the KNN classifier
def run_knn_trials(categories, ntrials=15, size=(32,32), test_size=0.25):
    print(f"Categories: {categories}")
    print(f"Resizing images to {size}, {ntrials} trials, test proportion {test_size}")
    
    acc_1nn = []
    acc_3nn = []
    acc_5nn = []
    acc_9nn = []
    for i in range(ntrials):
        acc = test_knn(categories, size, test_size)
        acc_1nn.append(acc[0])
        acc_3nn.append(acc[1])
        acc_5nn.append(acc[2])
        acc_9nn.append(acc[3])
        
    print(f"Average accuracy of 1NN: {round(np.mean(acc_1nn), 6)}")
    print(f"Average accuracy of 3NN: {round(np.mean(acc_3nn), 6)}")
    print(f"Average accuracy of 5NN: {round(np.mean(acc_5nn), 6)}")
    print(f"Average accuracy of 9NN: {round(np.mean(acc_9nn), 6)}")

# Trials for the Midterm Report
categories = ['069.fighter-jet','092.grapes','113.hummingbird','202.steering-wheel']
run_knn_trials(categories)
run_knn_trials(categories, test_size=0.2)
# Additional trials, results consistent with trials for image resizing to 32 x 32 pixels
# run_knn_trials(categories, size=(50,50))
# run_knn_trials(categories, size=(50,50), test_size=0.2)
# run_knn_trials(categories, size=(100,100))
# run_knn_trials(categories, size=(100,100), test_size=0.2)
# run_knn_trials(categories, size=(125,125))
# run_knn_trials(categories, size=(125,125), test_size=0.2)
# run_knn_trials(categories, size=(150,150))
# run_knn_trials(categories, size=(150,150), test_size=0.2)

# Repeat until enough images created:
# -> Select couple of images from same class (no matter which class, but same)
# -> Create synthetic image from them
# -> Save synthetic image with corresponding label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import morphing as mp
import cv2 as cv
import evaluator as ev
import data as dt



INDEX_PATH = 'train.csv'
TRAIN_PATH = 'train/'
SAVE_PATH = 'synthetic/'
N_SYNTHETIC = 1000

for i in range(N_SYNTHETIC):
    print(i)
    # Select couple of images
    df = pd.read_csv(INDEX_PATH, sep=';', names=['id', 'class'])
    r = np.random.randint(0,10)
    couple = df[df['class'] == r].sample(2).values.tolist()  # We have our couple
    p1 = TRAIN_PATH + str(couple[0][0]) + '-' +  str(couple[0][1]) + '.png'
    p2 = TRAIN_PATH + str(couple[1][0]) + '-' +  str(couple[1][1]) + '.png'

    # Read images and create synthetic one
    img0 = cv.imread(p1)
    img1 = cv.imread(p2)
    model = mp.FeaturesMorphing(img0, img1)
    model.create_synthetic_image()

    # Save synthetic image
    plt.imsave(SAVE_PATH + str(i) + '-' + str(r), model.synthetic_img, cmap='gray')


test_images = []
test_labels = []
evaluator = ev.Evaluator()
synthetic_images, synthetic_labels = dt.load_synthetic_images()
evaluator.evaluate(synthetic_images, synthetic_labels)

"""
for digit in range(0, 10):
    print(digit)
    images = []
    labels = []
    for i in range(0, len(synthetic_images)):
        if synthetic_labels[i] == digit:
            images.append(synthetic_images[i])
            labels.append(digit)
    evaluator.evaluate(np.array(images), np.array(labels))"""
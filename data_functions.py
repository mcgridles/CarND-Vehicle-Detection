import os
import glob
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

def loadDataset():
    datasets = [
        {
            'directory': 'object-dataset',
            'names': ['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'occluded', 'label', 'attributes'],
            'car_label': 'car',
            'sep': ' '
        },
        {
            'directory': 'object-detection-crowdai',
            'names': ['xmin', 'ymin', 'xmax', 'ymax', 'frame', 'label', 'preview'],
            'car_label': 'Car',
            'sep': None
        }
    ]

    for data in datasets:
        annotation_root = os.path.join('annotations', data['directory'])
        annotation_path = os.path.join(annotation_root, 'labels.csv')

        df = pd.read_csv(annotation_path, sep=data['sep'], header=None, names=data['names'])

        # filter out non-cars
        cars = df[df.label == data['car_label']]
        try:
            cars = cars.drop(['label', 'attributes', 'occluded'], 1)
        except ValueError:
            cars = cars.drop(['label', 'preview'], 1)

        cars = cars.groupby(['frame'], as_index = False)
        cars = cars.aggregate(lambda x : list(x))

        cars.reset_index()
        cars['frame'] = cars['frame'].apply(lambda x: os.path.join(annotation_root, x))

        yield cars

def carGenerator(sample_size=None):
    print('\033[1mLoading car dataset...\033[0m')

    # load GTI/KITTI datasets
    gti_df = pd.DataFrame(columns=['frame', 'xmin', 'ymin', 'xmax', 'ymax'])
    for vehicle_dir in os.listdir('annotations/vehicles'):
        vehicle_dir = os.path.join('annotations/vehicles', vehicle_dir)
        if os.path.isdir(vehicle_dir):
            frames = glob.glob(os.path.join(vehicle_dir, '*.png'))
            xmin = [[0]] * len(frames)
            ymin = [[0]] * len(frames)
            xmax = [[64]] * len(frames)
            ymax = [[64]] * len(frames)
            data = pd.DataFrame({'frame': frames, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            gti_df = gti_df.append(data)

    generator = loadDataset()
    autti_df = next(generator)
    crowdai_df = next(generator)

    # Reduce the sample size if necessary
    if sample_size:
        autti_df = autti_df.sample(frac=1).reset_index(drop=True)
        crowdai_df = crowdai_df.sample(frac=1).reset_index(drop=True)
        gti_df = gti_df.sample(frac=1).reset_index(drop=True)
        autti_df = autti_df[:sample_size]
        crowdai_df = autti_df[:sample_size]
        gti_df = gti_df[:sample_size]

    cars_df = pd.concat([gti_df, autti_df, crowdai_df])

    for i, row in cars_df.iterrows():
        filename = row['frame']
        img = mpimg.imread(filename)
        if filename[-3:] == 'jpg':
            img = img.astype(np.float32)/255

        for j in range(len(row['xmin'])):
            if row['ymin'][j] == row['ymax'][j] or row['xmin'][j] == row['xmax'][j]:
                continue

            # crop image to correct section and resize
            img_tosearch = img[int(row['ymin'][j]):int(row['ymax'][j]), int(row['xmin'][j]):int(row['xmax'][j]), :].astype(np.float32)
            img_tosearch = cv2.resize(img_tosearch, (64, 64))
            yield img_tosearch

def notCarGenerator(sample_size=None):
    print('\033[1mLoading non-car dataset...\033[0m')

    # load non-car dataset
    not_cars = []
    for non_vehicle_dir in os.listdir('annotations/non-vehicles'):
        non_vehicle_dir = os.path.join('annotations/non-vehicles', non_vehicle_dir)
        if os.path.isdir(non_vehicle_dir):
            not_cars.extend(glob.glob(os.path.join(non_vehicle_dir, '*.png')))

    if sample_size:
        not_cars = shuffle(not_cars)
        not_cars = not_cars[0:sample_size]

    # Functions for augmenting dataset
    basic_image = lambda image: image
    flip_image = lambda image: cv2.flip(image, 1)
    augmentation_funcs = [basic_image, flip_image]

    for filename in not_cars:
        img = mpimg.imread(filename)

        for func in augmentation_funcs:
            augmented_img = func(img)
            if np.max(img) > 1:
                augmented_img = img.astype(np.float32) / 255
                
            yield augmented_img

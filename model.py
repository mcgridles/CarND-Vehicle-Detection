import os
import numpy as np

from helper_functions import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

class Model(object):

    def __init__(self, c_space, orient, ppc, cpb, hog_chan, spatial_s, hist_b, spatial_f, hist_f, hog_f, force=False):
        # Using > 2858 images causes "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."
        self.sample_size = 2858

        self.color_space = c_space # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient  # HOG orientations
        self.pix_per_cell = ppc # HOG pixels per cell
        self.cell_per_block = cpb # HOG cells per block
        self.hog_channel = hog_chan # Can be 0, 1, 2, or "ALL"
        self.spatial_size = spatial_s # Spatial binning dimensions
        self.hist_bins = hist_b    # Number of histogram bins
        self.spatial_feat = spatial_f # Spatial features on or off
        self.hist_feat = hist_f # Histogram features on or off
        self.hog_feat = hog_f # HOG features on or off

        self.force = force

        self.X_scaler = None
        self.clf = LinearSVC(C=0.001)

    def train(self, cars, not_cars):
        # Reduce the sample size if necessary
        if self.sample_size:
            cars = cars[0:self.sample_size]
            not_cars = not_cars[0:self.sample_size]

        # Load classifier and data or generate it
        X_test, y_test = self.loadClassifier(cars, not_cars)
        score = round(self.clf.score(X_test, y_test), 4)
        print('\033[1mTest Accuracy of classifier:\033[92m', score, '\033[0m')

    def predict(self, features):
        return self.clf.predict(features)

    def generateClassifier(self, cars, not_cars):
        car_features = extractFeatures(cars, color_space=self.color_space,
                                       spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                       orient=self.orient, pix_per_cell=self.pix_per_cell,
                                       cell_per_block=self.cell_per_block,
                                       hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                       hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        not_car_features = extractFeatures(not_cars, color_space=self.color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, not_car_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        self.clf.fit(X_train, y_train)
        self.storeClassifier(X_test, y_test)

        return (X_test, y_test)

    def storeClassifier(self, X_test, y_test):
        joblib.dump((self.clf, self.X_scaler, X_test, y_test), 'clf.pkl')

    def loadClassifier(self, cars, not_cars):
        if not os.path.exists('clf.pkl') or self.force:
            return self.generateClassifier(cars, not_cars)
        else:
            loaded_data = joblib.load('clf.pkl')
            self.clf = loaded_data[0]
            self.X_scaler = loaded_data[1]
            X_test = loaded_data[2]
            y_test = loaded_data[3]
            return (X_test, y_test)

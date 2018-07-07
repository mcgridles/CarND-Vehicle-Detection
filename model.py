import numpy as np

from helper_functions import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class Model(object):

    def __init__(self):
        self.sample_size = 500

        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 1 # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16) # Spatial binning dimensions
        self.hist_bins = 16    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = False # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.y_start_stop = [360, 720] # Min and max in y to search in slide_window()

        self.classifier = LinearSVC()

    def train(self, cars, notcars):
        # Reduce the sample size if necessary
        cars = cars[0:self.sample_size]
        notcars = notcars[0:self.sample_size]

        car_features = extract_features(cars, color_space=self.color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        # Use a linear SVC
        self.classifier.fit(X_train, y_train)
        score = round(self.classifier.score(X_test, y_test), 4)
        print('Test Accuracy of SVC = ', score)

    def predict(self):
        pass

import os
import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from helper_functions import *
from model import Model
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

class Detector(object):

    def __init__(self):
        self.x_start = 0
        self.x_stop = 1280
        self.y_start = 380
        self.y_stop = 656
        self.scale = 1.5
        self.xy_window = (128, 128)
        self.xy_overlap = (0.5, 0.5)

        self.color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 3 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 16    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off

        self.clf = Model(
            self.color_space,
            self.orient,
            self.pix_per_cell,
            self.cell_per_block,
            self.hog_channel,
            self.spatial_size,
            self.hist_bins,
            self.spatial_feat,
            self.hist_feat,
            self.hog_feat
        )
        self.initModel()

    def initModel(self):
        cars = []
        not_cars = []
        for vehicle_dir in os.listdir('annotations/vehicles'):
            vehicle_dir = os.path.join('annotations/vehicles', vehicle_dir)
            if os.path.isdir(vehicle_dir):
                cars.extend(glob.glob(os.path.join(vehicle_dir, '*.png')))

        for non_vehicle_dir in os.listdir('annotations/non-vehicles'):
            non_vehicle_dir = os.path.join('annotations/non-vehicles', non_vehicle_dir)
            if os.path.isdir(non_vehicle_dir):
                not_cars.extend(glob.glob(os.path.join(non_vehicle_dir, '*.png')))
        self.clf.train(cars, not_cars)

    def detectCars(self, img):
        img = img.astype(np.float32)/255

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=[self.y_start, self.y_stop],
                               xy_window=self.xy_window, xy_overlap=self.xy_overlap)

        hot_windows = self.search_windows(img, windows)
        # hot_windows = self.find_cars(img)

        # create heatmap
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heat = apply_threshold(heat, 1)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)

        boxes = draw_labeled_bboxes(np.copy(img), labels)
        boxes = draw_boxes(img, hot_windows)
        plt.imshow(boxes)
        plt.show()

    def single_img_features(self, img):
        img_features = []
        if self.color_space != 'RGB':
            feature_image = convert_color(img, 'RGB2' + self.color_space)
        else:
            feature_image = np.copy(img)

        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            img_features.append(spatial_features)

        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            img_features.append(hist_features)

        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                        self.orient, self.pix_per_cell, self.cell_per_block,
                                        vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], self.orient, self.pix_per_cell,
                                                self.cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

        return np.concatenate(img_features)

    def search_windows(self, img, windows):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.single_img_features(test_img)

            test_features = self.clf.X_scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.clf.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def find_cars(self, img):
        img_tosearch = img[self.y_start:self.y_stop,:,:]
        img_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/self.scale), np.int(imshape[0]/self.scale)))

        ch1 = img_tosearch[:,:,0]
        ch2 = img_tosearch[:,:,1]
        ch3 = img_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        on_windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                features = []
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step

                # Extract HOG for this patch
                if self.hog_feat:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                if self.spatial_feat:
                    spatial_features = bin_spatial(subimg, size=self.spatial_size)
                    features = np.hstack((features, spatial_features))

                if self.hist_feat:
                    hist_features = color_hist(subimg, nbins=self.hist_bins)
                    features = np.hstack((features, hist_features))

                # Scale features and make a prediction
                test_features = self.clf.X_scaler.transform(features.reshape(1, -1))
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * self.scale)
                    ytop_draw = np.int(ytop * self.scale)
                    win_draw = np.int(window * self.scale)
                    on_windows.append((
                        (xbox_left, ytop_draw+self.y_start),
                        (xbox_left+win_draw,ytop_draw+win_draw+self.y_start)
                    ))

        return on_windows

if __name__ == '__main__':
    img = mpimg.imread('test_images/test1.jpg')
    detector = Detector()
    detector.detectCars(img)

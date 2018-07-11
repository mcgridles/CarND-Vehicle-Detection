import os
import cv2
import sys
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
from collections import deque
from moviepy.editor import VideoFileClip

class Detector(object):

    def __init__(self, force):
        self.x_start = 0
        self.x_stop = 1280
        self.y_start = 350
        self.y_stop = 656
        self.scale = 1
        self.xy_window = (64, 64)
        self.xy_overlap = (0.5, 0.5)

        self.color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 11  # HOG orientations
        self.pix_per_cell = 16 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32) # Spatial binning dimensions
        self.hist_bins = 32    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.heat_threshold = 10

        self.heatmaps_threshold = 5
        self.heatmaps = deque(maxlen=self.heatmaps_threshold)

        self.classify = Model(
            self.color_space,
            self.orient,
            self.pix_per_cell,
            self.cell_per_block,
            self.hog_channel,
            self.spatial_size,
            self.hist_bins,
            self.spatial_feat,
            self.hist_feat,
            self.hog_feat,
            force
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
        self.classify.train(cars, not_cars)

    def detectCars(self, img):
        hot_windows = self.findCars(img)

        # create heatmap
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = addHeat(heat, hot_windows)
        heat = applyThreshold(heat, 3)
        self.heatmaps.append(heat)
        if len(self.heatmaps) == 0:
            prev_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        else:
            prev_heatmap = sum(self.heatmaps)
        agg_heatmap = prev_heatmap + heat
        thresh_heatmap = applyThreshold(agg_heatmap, self.heat_threshold)
        thresh_heatmap = np.clip(thresh_heatmap, 0, 255)
        labels = label(thresh_heatmap)

        boxes = drawLabeledBoxes(np.copy(img), labels)
        # plt.imshow(thresh_heatmap, cmap='hot')
        # plt.imshow(boxes)
        # plt.show()
        return boxes

    def singleImgFeatures(self, img):
        img_features = []
        if self.color_space != 'RGB':
            feature_image = convertColor(img, 'RGB2' + self.color_space)
            img = img.astype(np.float32)/255
        else:
            feature_image = np.copy(img)

        if self.spatial_feat == True:
            spatial_features = binSpatial(feature_image, size=self.spatial_size)
            img_features.append(spatial_features)

        if self.hist_feat == True:
            hist_features = colorHist(feature_image, nbins=self.hist_bins)
            img_features.append(hist_features)

        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(getHogFeatures(feature_image[:,:,channel],
                                        self.orient, self.pix_per_cell, self.cell_per_block,
                                        vis=False, feature_vec=True))
            else:
                hog_features = getHogFeatures(feature_image[:,:,hog_channel], self.orient, self.pix_per_cell,
                                                self.cell_per_block, vis=False, feature_vec=True)
            img_features.append(hog_features)

        return np.concatenate(img_features)

    def searchWindows(self, img, windows):
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = self.singleImgFeatures(test_img)

            test_features = self.classify.X_scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.classify.predict(test_features)
            if prediction == 1:
                on_windows.append(window)
        return on_windows

    def findCars(self, img):
        img_tosearch = img[self.y_start:self.y_stop,:,:].astype(np.float32)/255
        img_tosearch = convertColor(img_tosearch, 'RGB2' + self.color_space)
        img = img.astype(np.float32)/255
        if self.scale != 1:
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
        hog1 = getHogFeatures(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = getHogFeatures(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = getHogFeatures(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        on_windows = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                features = []
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                if self.hog_feat:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                if self.spatial_feat:
                    spatial_features = binSpatial(subimg, size=self.spatial_size)
                    features = np.hstack((features, spatial_features))

                if self.hist_feat:
                    hist_features = colorHist(subimg, nbins=self.hist_bins)
                    features = np.hstack((features, hist_features))

                # Scale features and make a prediction
                test_features = self.classify.X_scaler.transform(features.reshape(1, -1))
                test_prediction = self.classify.predict(test_features)

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
    try:
        # Force train new model
        if sys.argv[1] == 'force':
            force = True
        else:
            force = False
    except IndexError:
        force = False

    detector = Detector(force)
    
    clip = VideoFileClip('project_video.mp4')
    processed_video = clip.fl_image(detector.detectCars).subclip(10, 13)
    processed_video.write_videofile('output_video.mp4', audio=False)

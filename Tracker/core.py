__author__ = 'pieter'

import collections
import time

import numpy as np
import scipy as sp

import os
import csv
import json

import video

import cv2    # '2.4.6.1'
from itertools import permutations
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import socket

class Detector:
    def __init__(self, config):
        # Get all the settings from the config file
        self.video_folder = config.get('default', 'video_folder')
        self.video_files = json.loads(config.get('default', 'video_files'))
        self.visualize = json.loads(config.get('default', 'visualize'))
        self.number_of_objects = json.loads(config.get('default', 'number_of_objects'))
        self.inertia_threshold = json.loads(config.get('default', 'inertia_threshold'))
        self.arena_settings = json.loads(config.get('default', 'arena_settings'))
        self.led_settings = json.loads(config.get('default', 'led_settings'))
        self.lk_settings = json.loads(config.get('default', 'lk_settings'))
        self.detector = json.loads(config.get('default', 'detector'))
        self.ShiTom_settings = json.loads(config.get('default', 'ShiTom_settings'))
        self.FAST_settings = json.loads(config.get('default', 'FAST_settings'))

        self.arena_settings["points"] = np.array(self.arena_settings["points"])
        self.led_settings["center_1"] = tuple(self.led_settings["center_1"])
        self.led_settings["center_2"] = tuple(self.led_settings["center_2"])

        self.lk_settings["winSize"] = tuple(self.lk_settings["winSize"])
        self.lk_settings["criteria"] = (long(3), self.lk_settings["criteria_count"], self.lk_settings["criteria_eps"])
        del self.lk_settings["criteria_eps"], self.lk_settings["criteria_count"]

        try:
            self.mmse_lookback = json.loads(config.get('default', 'MMSE_lookback'))
        except:
            self.mmse_lookback = 6

        self.permutations = np.array(list(permutations(range(self.number_of_objects))))
        self.color_pallet = plt.get_cmap('jet')(np.linspace(0, 1.0, self.number_of_objects))*255

        self.track_len = 10  # must be higher than 5
        self.tracks = []

        self.visual_image = None

        self.frame_idx = 1
        self.start_time = 0.

        self.previous_centers = None
        self.ordered_centers = None
        self.new_labels = None

        self.led_status = [0, 0]
        self.led_values = [0, 0]

        self.arena_mask = None

        print(socket.gethostname())
        if socket.gethostname() == "lubuntu":
            print("Hey, Richel's computer is detected")
            self.fast = cv2.FastFeatureDetector_create(**self.FAST_settings)
        else:
            print("Use OpenCV 2.4's version")
            self.fast = cv2.FastFeatureDetector_create(**self.FAST_settings)

        #self.fgbg = cv2.BackgroundSubtractorMOG()

    def _init_bounding_box(self):
        # Get the bounding box, [minX, minY] and [maxX, maxY]
        self.min_xy = np.min(self.arena_settings["points"], 0)
        self.max_xy = np.max(self.arena_settings["points"], 0)

    def _init_data_writer(self, video_file):
        """
        - Initialize the data writer.
        - Write the header to the csv output file

        :param video_file: video file name
        :return:
        """
        # Setup a csv writer
        output_file = self.video_folder + video_file[:video_file.find('.')]+'_output.csv'
        self.output_csv = open(output_file, "wb")
        self.data_writer = csv.writer(self.output_csv, delimiter=',')

        # Write a header
        header = ["frame_idx", "led_1_status", "led_2_status"]
        for i in range(self.number_of_objects):
            header.extend((str(i) + "_x", str(i) + "_y"))
        self.data_writer.writerow(header)

    def _init_video_writer(self, video_file):
        """
        - Setup video reader
        - Initialize the video writer

        :param video_file: video file name
        :return:
        """

        # Setup a frame reader
        print "I am going to open file '" + self.video_folder + video_file + "'"
        if os.path.isfile(self.video_folder + video_file) == False:
                print "Error: cannot find file '" + self.video_folder + video_file + "'"
                print "Tip: do 'wget http://richelbilderbeek.nl/3f_1.mp4' or './download_video'"
                raise SystemExit

        print("File is present: " + str(os.path.isfile(self.video_folder + video_file)))
        self.camera = video.create_capture(self.video_folder+video_file)

        # Setup video writer
        self.video_writer = cv2.VideoWriter()

        # Setup the video output paths and names
        self.output_video = self.video_folder + video_file[:video_file.find('.')]+'_output.avi'

        # Setup a video writer with the size of the bounding box
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.video_writer.open(self.output_video, four_cc, 100, (self.max_xy[0]-self.min_xy[0],
                                                                 self.max_xy[1]-self.min_xy[1]), True)

        self.total_number_frames = int(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def _init_k_means(self):
        """
        - Initialize the random k-means
        - Initialize the k-means with predefined seeds

        :return:
        """
        # Init random k-means
        self.k_means_random = KMeans(n_clusters=self.number_of_objects, init='random', n_init=10, max_iter=1000)
        # call the fit function so that it has .cluster_centers_
        self.k_means_random.fit([[0]]*self.number_of_objects)

        # Init k-means with ordered centers as seed init.
        self.k_means = KMeans(n_clusters=self.number_of_objects, init='k-means++', n_init=1, max_iter=10000,
                              tol=0.00000000001)
        # call the fit function so that it has .cluster_centers_
        self.k_means.fit([[0]]*self.number_of_objects)

    def read_frame(self, video_file):
        ret, frame = self.camera.read()

        if not ret:
            if self.frame_idx > self.total_number_frames:
                # Video is finished
                print "Finished file: {1}".format(video_file)
                return None
            else:
                # Skip this bad frame
                print "Error({0}): {1}".format("couldn't read frame", self.frame_idx)
                return None

        return frame

    def create_arena_mask(self, frame):
        # Create a mask with the size of the original camera frame
        self.arena_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Color the mask white in ROI(Arena)
        cv2.fillPoly(self.arena_mask, [self.arena_settings["points"]], color=255)

        # Cut the mask with ROI(Bounding box)
        self.arena_mask = self.roi_bounding_box(self.arena_mask)

    def background_threshold(self, image):
        image[image > self.arena_settings['background_threshold']] = 0
        return image

    def background_kmeans(self, image):
        # make a list of value instead of a 2d array
        data = image.reshape((image.shape[0] * image.shape[1], 1))
        #self.k_means_bg.fit(data)
        #print self.k_means_bg.cluster_centers_

    def background_subtractor(self, image):
        fgmask = self.fgbg.apply(image)
        return cv2.bitwise_and(image, image, mask=fgmask)

    def apply_mask(self, image):
        return cv2.bitwise_and(image, image, mask=self.arena_mask)

    def roi_bounding_box(self, image):
        return image[self.min_xy[1]:self.max_xy[1], self.min_xy[0]:self.max_xy[0]]

    def draw_circles(self, image, points):

        [cv2.circle(image, xy, 1, self.color_pallet[label], -1) for xy, label in zip(points, self.new_labels)]

        # From float to int
        centers = self.ordered_centers.astype(int)
        [cv2.circle(image, tuple(xy), 4, color, -1) for color, xy in zip(self.color_pallet, centers)]

    def visualizer(self, video_file):
        cv2.imshow("TRACKING VISUALISER for video file: {0}".format(video_file), self.visual_image)

    def write_data(self):
        if self.ordered_centers is None:
            return
        coordinates = self.ordered_centers.astype(int).flatten().tolist()
        self.data_writer.writerow([self.frame_idx, self.led_status[0], self.led_status[1]] + coordinates)

    def stop(self, force=False):
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27 or force:
            cv2.destroyAllWindows()
            self.output_csv.close()
            return True
        return False

    def debug(self, frame, video_file):
        frame_copy = frame.copy()

        if self.arena_settings["debug"]:
            cv2.polylines(frame_copy, [self.arena_settings["points"]], 1, color=255)

        if self.led_settings["debug"]:
            cv2.circle(frame_copy, self.led_settings["center_1"], self.led_settings["radius_1"]*2, (0, 255, 0), 1)
            cv2.circle(frame_copy, self.led_settings["center_2"], self.led_settings["radius_2"]*2, (0, 255, 0), 1)

        # Show it on the screen
        cv2.imshow("Debug window for file: {0}".format(video_file), frame_copy)
        cv2.waitKey(0)
        cv2.destroyWindow("Debug window for file: {0}".format(video_file))

    def get_led_status(self, image):

        # Convert to Gray-scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        radius_led_1 = self.led_settings["radius_1"]
        [x_led_1, y_led_1] = self.led_settings["center_1"]

        radius_led_2 = self.led_settings["radius_2"]
        [x_led_2, y_led_2] = self.led_settings["center_2"]

        # Cut the region of interest first because it's faster
        roi_led_1 = image[y_led_1-radius_led_1:y_led_1+radius_led_1, x_led_1-radius_led_1:x_led_1+radius_led_1]
        roi_led_2 = image[y_led_2-radius_led_2:y_led_2+radius_led_2, x_led_2-radius_led_2:x_led_2+radius_led_2]

        # Setup a mask image
        mask_led_1 = np.zeros(shape=roi_led_1.shape, dtype=np.uint8)
        mask_led_2 = np.zeros(shape=roi_led_2.shape, dtype=np.uint8)

        # Creating the mask
        cv2.circle(mask_led_1, (radius_led_1, radius_led_1), radius_led_1*2, 255, -1)
        cv2.circle(mask_led_2, (radius_led_2, radius_led_2), radius_led_2*2, 255, -1)

        # Apply the mask to the gray-scale image
        resulting_image_led_1 = cv2.bitwise_and(roi_led_1, roi_led_1, mask=mask_led_1)
        resulting_image_led_2 = cv2.bitwise_and(roi_led_2, roi_led_2, mask=mask_led_2)

        # Get the mean led value
        self.led_values[0] = resulting_image_led_1[resulting_image_led_1 > 0].mean()
        self.led_values[1] = resulting_image_led_2[resulting_image_led_2 > 0].mean()

        # Apply threshold
        self.led_status[0] = int(self.led_values[0] > self.led_settings["threshold_left"])
        self.led_status[1] = int(self.led_values[1] > self.led_settings["threshold_right"])

    def detect_features(self, gray_image, previous_gray_image):
        """
         - This function detects features with the openCV's goodFeaturesToTrack. Then these features are tracked from one
         frame to the next with the openCV's Lucas Kanade optical flow method.
         - Then features are detected again using the goodFeaturesToTrack method, but this time only where there are no
         features already. This is done using a mask.
        :param gray_image: the pre-processed gray image of the arena
        :param previous_gray_image: the pre-processed gray image of the previous frame
        :return: the newest added point from each track
        """

        # Make sure there are features to track
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks])  # .reshape(-1, 1, 2)

            # Track all features forward
            p1, st, err = cv2.calcOpticalFlowPyrLK(previous_gray_image, gray_image, p0, None, **self.lk_settings)
            # Track all features from the previous algorithm backwards.
            p0r, st, err = cv2.calcOpticalFlowPyrLK(gray_image, previous_gray_image, p1, None, **self.lk_settings)

            # Now we can use the forward-backward error to eliminate the bad matches
            d = np.abs(np.subtract(p0, p0r)).reshape(-1, 2).max(-1)
            good = d < 1.0

            # Create a new array of tracks
            new_tracks = []

            # Add tracks to new array if the match is good
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                # Means that forward-backwards error is high, so not a good match, let's skip it
                if not good_flag:
                    continue

                # Else we add the new point
                tr.append((x, y))

                # Prevent the tracks from growing to long
                if len(tr) > self.track_len:
                    del tr[0]

                # Finally we want to collect the good tracks
                new_tracks.append(tr)

            # Let's overwrite the old set of tracks with the new set of tracks
            self.tracks = new_tracks

            cv2.putText(self.visual_image, "Features:%d" % len(self.tracks), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0))

        # Creating a mask to prevent the goodFeaturesToTrack algorithm to detect features that have already been
        # detected.
        mask = np.zeros_like(gray_image)
        mask[:] = 255

        # Fill the mask with black(0) at the features positions, so that it doesn't find any features there.
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 10, 0, -1)

        if self.detector["ShiTomasi"] == True:
            # Track features with the goodFeaturesToTrack algorithm
            p = cv2.goodFeaturesToTrack(gray_image, mask=mask, **self.ShiTom_settings)

        elif self.detector["FAST"] == True:
            # Track features with the FAST feature detector
            p = self.fast.detect(gray_image, mask=mask)
            # We only need the coordinates(pt)
            p = np.array([point.pt for point in p])
        else:
            print "No feature detector is selected"
            p = None

        # Append new features to the tracks array
        if p is not None:
            for xy in p.reshape(-1, 2):
                self.tracks.append([tuple(xy)])

        # We return only points that have a track history of > 5, this to have stable features.
        return [tuple(list(tr[-1])) for tr in self.tracks if len(tr) > 5], gray_image

    def cluster_points(self, points):
        #
        if len(points) >= self.number_of_objects:

            # At the start we do a random initialization, since we don't have any information.
            if len(self.previous_centers) == 0:
                self.k_means_random.fit(points)
                self.new_labels = self.k_means_random.labels_
                self.ordered_centers = self.k_means_random.cluster_centers_
            else:
                if len(points) == 1:
                    self.k_means_random.cluster_centers_ = np.array(points)
                    self.k_means_random.labels_ = [0]
                    self.k_means_random.inertia_ = 0
                else:
                    # Apply the k-means with random init.
                    self.k_means_random.fit(points)

                if len(points) == 1:
                    self.k_means.cluster_centers_ = np.array(points)
                    self.k_means.labels_ = [0]
                    self.k_means.inertia_ = 0
                else:
                    # Apply the k-means with previous centers as init
                    self.k_means.set_params(init=self.ordered_centers)
                    self.k_means.fit(points)

                # check if the k-means is drifting from its target
                if self.k_means.inertia_ > self.inertia_threshold * self.k_means_random.inertia_:
                    self.mmse(self.previous_centers, self.k_means_random.cluster_centers_, self.k_means_random.labels_)
                else:
                    self.ordered_centers = self.k_means.cluster_centers_
                    self.new_labels = self.k_means.labels_


            # Draw circles for features and centers
            self.draw_circles(self.visual_image, points)

            # We append because previous centers is a collection with a fixed length
            self.previous_centers.append(self.ordered_centers)

    def calc_fps(self):
        # Calculate FPS
        time_taken = time.time() - self.start_time
        fps = int(self.frame_idx/time_taken)
        cv2.putText(self.visual_image, "FPS:%i" % fps, (125, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Calculate estimated time it takes
        minutes_to_go = int((self.total_number_frames-self.frame_idx)/(self.frame_idx/time_taken)/60)

        cv2.putText(self.visual_image, "Remaining:%im" % minutes_to_go, (200, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0))

        cv2.putText(self.visual_image, "Frame:%i" % self.frame_idx, (350, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0))

        #Andreas addition
        cv2.putText(self.visual_image, "LED:%i %i" % (self.led_status[0], self.led_status[1]), (495, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0))
        #Andreas addition
        cv2.putText(self.visual_image, "V:%i/%i %i/%i" % (self.led_values[0], self.led_settings["threshold_left"], self.led_values[1], self.led_settings["threshold_right"]), (570, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0))
        self.frame_idx += 1

        return fps

    def mmse(self, previous_centers_collection, centers, labels):
        """
         This function calculates the Least Square Error for each permutation of centers compared to the
         previous centers. Like K-NN but than nicer. because it calculates the k-NN for all points at the same
         time. The problem with this algorithm is that it is number_of_objects!
        """

        labels = labels.flatten()
        best_option_error = np.inf
        trans_array = np.arange(self.number_of_objects)

        # Defaults
        best_option = centers
        new_labels = labels
        print "MMSE applied @ frame:", self.frame_idx
        # Loop trough previous centers collection, the one with the lowest error is probably the best.
        # We do this because the last assignment is probably not correct, because it is drifting. So we check a couple
        # of time step back to check if we get a lower error value.
        remember = previous_centers_collection.pop()  # TODO: check if this is correct
        for previous_centers in previous_centers_collection:
            if previous_centers is None:
                continue

            # This calculates a 2D array with euclidean distances from each point in prev_centers to each other point
            # in centers.
            distance_map = sp.spatial.distance.cdist(previous_centers, centers, 'euclidean')

            # This returns a list with summed distances(error) for each permutation of center assignments.
            error_list = distance_map[trans_array, self.permutations].sum(1)

            # Get the index of the lowest error
            index = np.argmin(error_list)

            # Get the error value
            option_error = error_list[index]

            # Get the right center assignment
            best_trans = self.permutations[index]

            # Smaller error is better
            if option_error < best_option_error:
                best_option_error = option_error
                best_option = centers[best_trans]
                new_labels = best_trans[labels]

        self.ordered_centers = np.array(best_option)
        self.new_labels = new_labels

        previous_centers_collection.append(remember)

    def run(self):
        for video_file in self.video_files:

            # Reset some values
            self.frame_idx = 1
            self.previous_centers = []
            self.ordered_centers = None
            self.tracks = []
            previous_gray_image = []

            # init
            self._init_bounding_box()
            self._init_video_writer(video_file)
            self._init_data_writer(video_file)
            self._init_k_means()

            try:
                # Read the next frame from the video file
                frame = self.read_frame(video_file)
            except:
                print "Error({0}): {1}".format("couldn't find file", video_file)
                continue

            # Go directly to debug mode if debug == true
            if self.arena_settings["debug"] or self.led_settings["debug"]:
                self.debug(frame, video_file)
                continue

            if self.number_of_objects > 6 or self.number_of_objects == 1:
                # Because above 6 objects the MMSE algorithm starts to get
                # slow we don't look back more than one step.
                self.previous_centers = collections.deque(maxlen=1)
            else:
                # Here we look back 6 steps in time with the MMSE algorithm
                self.previous_centers = collections.deque(maxlen=self.mmse_lookback)

            self.create_arena_mask(frame)

            self.start_time = time.time()

            # Run
            while True:
                # Read next frame
                frame = self.read_frame(video_file)
                if frame is None:
                    self.stop(force=True)
                    break

                # Convert to gray scale
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Get Led status
                self.get_led_status(frame)

                # Roi the image
                gray_image = self.roi_bounding_box(gray_image)

                # Apply the mask to the gray scale image
                gray_image = self.apply_mask(gray_image)

                self.background_kmeans(gray_image)

                # Apply the threshold, this is all values above the threshold make black
                gray_image = self.background_threshold(gray_image)

                # Create a video image to view and save on disk
                self.visual_image = self.roi_bounding_box(frame)

                # Do the feature detection
                points, previous_gray_image = self.detect_features(gray_image, previous_gray_image)

                # Cluster the points
                self.cluster_points(points)

                # Write the coordinates to the output file
                self.write_data()

                # Calculate FPS
                fps = self.calc_fps()

                # Write image to video
                self.video_writer.write(self.visual_image)

                # Visualize
                if self.visualize:
                    self.visualizer(video_file)

                # Quit if Esc-key is pressed
                if self.stop():
                    break


class Correction:
    def __init__(self, config):
        self.video_folder = config.get('default', 'video_folder')
        self.video_files = json.loads(config.get('default', 'video_files'))
        self.output_file = None

    def run(self):
        for video_file in self.video_files:
            ordered_data, data = self.get_data(video_file)

            if not ordered_data:
                print "File {0} is empty".format(video_file[:video_file.find('.')]+'_output.csv')
                continue

            for idx in range(np.shape(ordered_data)[1]):
                outliers, idx_data = self.detect_outliers(ordered_data, idx)
                data = self.correct(idx_data, data, outliers, idx)

            self.write_data(data)

    def _to_matrix(self, l, n):
        return [self._to_int(l[i:i+n]) for i in xrange(0, len(l), n)]

    def write_data(self, data):

        output_file = self.output_file[:-4]+'_corrected.csv'
        try:
            f = open(output_file, 'wt')  # opens the csv file
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            [writer.writerow(i) for i in data]
            f.close()
        except:
            print "Coulnd't write file {0}".format(output_file)

    def get_data(self, video_file):
        self.output_file = self.video_folder + video_file[:video_file.find('.')]+'_output.csv'
        output_csv = open(self.output_file, "rb")
        data = self.read_data_from_csv(output_csv)
        output_csv.close()      # closing
        return self.order_data(data), data

    def order_data(self, data):
        return [self._to_matrix(x[3:], 2) for x in data[1:]]

    @staticmethod
    def read_data_from_csv(f):
        try:
            reader = csv.reader(f)  # creates the reader object
            return [row for row in reader]
        except:
            print "Couldn't read file"
            return []

    @staticmethod
    def _to_int(l):
        if isinstance(l, list):
            return [int(el) for el in l]
        return int(l)

    @staticmethod
    def detect_outliers(data, idx):
        idx_data = np.array(data)[:, idx, :]

        x_1, y_1 = 0.0, 0.0
        dis_array = []
        for cor in idx_data:
            x, y = float(cor[0]), float(cor[1])
            # calculate the euclidean distance between each point to the next
            dis = np.sqrt((x-x_1)**2 + (y-y_1)**2)
            dis_array.append(dis)
            x_1, y_1 = x, y

        dis_array = np.array(dis_array)
        # get the points where the distance is > 50, these are the outliers
        outliers = np.where(dis_array > 50)[0]  #TODO: make it a parameter
        return outliers, idx_data

    @staticmethod
    def correct(idx_data, data, outliers, idx):
        skip_next = False
        for i, outlier in enumerate(outliers[:-1]):
            # skip the first 50 frames since the algorithm needs time to settle
            skip_start = outlier < 50
            # stop 5 frames before the end
            skip_end = outlier > len(data)-5

            # if the next outlier should be within 30 points from the
            # original outlier, else it is not considered as a deviation

            skip_out_of_range = outlier + 30 < outliers[i+1]

            # these 50 and

            # skip_next, this is when the deviation is already detected
            if skip_start or skip_end or skip_next or skip_out_of_range:
                # skip this outlier, then set skip_next to false
                skip_next = False
            else:
                # Starting point of the deviation
                start_out = outlier
                # the value of the point before deviation is detected
                start_value = idx_data[outlier-1]
                # i+1 so we have the next outlier which should be the return to the track,
                end_out = outliers[i+1]  # +1
                # the value of the point after it's returned to the track
                end_value = idx_data[end_out]  # end_out-1
                # get the number of point between the start and end point of the deviation
                diff = end_out - start_out  # -1

                # get the distance between the start and end point of the deviation for x and y
                x_diff = float(end_value[0] - start_value[0])
                y_diff = float(end_value[1] - start_value[1])
                # calculate the distance of one step
                x_diff /= (diff+1)
                y_diff /= (diff+1)

                # calculate the interpolated points using the above information
                points = [(int(start_value[0]+((point+1)*x_diff)), int(start_value[1]+((point+1)*y_diff)))
                          for point in range(diff)]

                # place the corrected points back into the data
                for j, point in enumerate(points):
                    #print data[start_out+j+1][idx*2+3]
                    data[start_out+j+1][idx*2+3] = point[0]
                    data[start_out+j+1][idx*2+4] = point[1]
                # we skip the next since we used that one to return to the track
                skip_next = True
        return data




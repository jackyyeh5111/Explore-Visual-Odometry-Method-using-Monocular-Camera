import os
import cv2
import numpy as np
import cv2
import os
from utils import *
from factories import FeatureDetectorFactory, FeatureMatcherFactory, MotionEstimatorFactory


class VisualOdometry(object):
    def __init__(self, args, gt_pose, img_file_path):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
        '''

        self.args = args
        args.K = np.array([[args.focal, 0.0, args.pp[0]],
                           [0.0, args.focal, args.pp[1]],
                           [0.0,        0.0,       1.0]])
        self.detector = FeatureDetectorFactory.generate(args)
        if not args.use_tracking:
            self.matcher = FeatureMatcherFactory.generate(args)

        self.motion_estimator = MotionEstimatorFactory.generate(args)
        self.gt_pose = gt_pose

        self.kptdes = {}
        self.file_path = img_file_path
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0
        self.img_shape = cv2.imread(
            self.file_path + str().zfill(6) + '.png', 0).shape

        self.process_frame()

    def detect(self, img):
        '''Used to detect features and parse into useable format
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        keypoints = np.array([x.pt for x in keypoints], dtype=np.float64)
        return {'keypoints': keypoints, 'descriptors': descriptors}

    def match(self):
        # feature detection
        if self.id == 0:
            self.kptdes['ref'] = self.detect(self.old_frame)
        else:
            self.kptdes['ref'] = self.kptdes['cur']
        self.kptdes['cur'] = self.detect(self.current_frame)

        # feature matching
        good_old, good_new = self.matcher.match(self.kptdes)
        return good_old, good_new

    def track(self):
        if self.n_features < 2000:
            # feature detection
            kps = self.detector.detect(self.old_frame)
            self.p0 = np.array([x.pt for x in kps],
                               dtype=np.float32).reshape(-1, 1, 2)

        lk_params = dict(winSize=(21, 21), criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # feature tracking
        # Calculate optical flow between frames, st holds status of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_frame, self.current_frame, self.p0, None, **lk_params)

        # Save the good points from the optical flow
        good_old = self.p0[st == 1]
        good_new = self.p1[st == 1]

        # Save the total number of good features
        self.n_features = good_new.shape[0]
        return good_old, good_new

    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''

        # ----- feature detection & feature matching/tracking -----
        get_good_feature = self.track if self.args.use_tracking else self.match
        good_old, good_new = get_good_feature()

        # ----- motion estimation -----
        # compute relative R, t between ref and cur frame
        R, t = self.motion_estimator.estimate(good_old, good_new)

        # If the frame is one of first two, we need to initalize our t and R vectors
        if self.id < 2:
            self.R = R
            self.t = t
        else:
            # get absolute pose based on absolute_scale
            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale * self.R @ t
                self.R = R @ self.R

    @property
    def pred_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)
        return adj_coord.flatten()

    @property
    def gt_coordinates(self):
        '''Returns true coordinates of vehicle
        Returns:
            np.array -- Array in format [x, y, z]
        '''
        return self.true_coord.flatten()

    def get_absolute_scale(self):
        '''Used to provide scale estimation for mutliplying
           translation vectors
        Returns:
            float -- Scalar value allowing for scale estimation
        '''
        pose = self.gt_pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.gt_pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        return np.linalg.norm(true_vect - prev_vect)

    def hasNextFrame(self):
        '''Used to determine whether there are remaining frames
           in the folder to process
        Returns:
            bool -- Boolean value denoting whether there are still 
            frames in the folder to process
        '''
        return self.id < len(os.listdir(self.file_path))

    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''
        if self.id < 2:
            self.old_frame = cv2.imread(
                self.file_path + str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(
                self.file_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(
                self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1

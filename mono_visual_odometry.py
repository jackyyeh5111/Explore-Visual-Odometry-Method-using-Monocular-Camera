import os
import cv2
import numpy as np
import cv2
import os
from optimizer import PoseGraphOptimization
from utils import *



class MonoVisualOdometery(object):
    def __init__(self, 
                 args,
                 img_file_path, 
                 pose_file_path, 
                 detector, 
                 matcher, 
                 focal_length, 
                 pp, 
                 ransac=5):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequence
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        '''
        if detector == 'FAST' and matcher != 'LK':
            raise ValueError(
                'FAST is not a keypoint descriptor and can only be used with Lucas–Kanade.')

        if detector == 'FAST':
            self.detector = cv2.FastFeatureDetector_create(
                threshold=25, nonmaxSuppression=True)
        elif detector == 'BRISK':
            self.detector = cv2.BRISK_create(thresh=25)
        elif detector == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=1000, fastThreshold=20)
        elif detector == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=2000)
        else:
            raise ValueError('Unknown detector type: {}'.format(detector))

        if matcher == 'LK':
            self.lk_params = dict(winSize=(21, 21), criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        elif matcher == 'BF':
            if detector == 'SIFT':
                self.matcher = cv2.BFMatcher()
            else:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher == 'FLANN':
            if detector == 'SIFT':
                index_params = dict(algorithm=1, trees=5)
            else:
                index_params = dict(algorithm=6, table_number=6,
                                    key_size=12, multi_probe_level=2)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError('Unknown matcher type: {}'.format(matcher))

        self.args = args
        self.ransac = ransac
        self.RANSAC_NUM_ITERATIONS = 25
        self.RANSAC_THRESHOLD = 1
        self.detector_name = detector
        self.matcher_name = matcher
        self.kptdes = {}
        self.file_path = img_file_path
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0
        self.img_shape = cv2.imread(
            self.file_path + str().zfill(6) + '.png', 0).shape
        self.K = np.array([[7.215377000000e02, 0.000000000000e00, 6.095593000000e02],
                           [0.000000000000e00, 7.215377000000e02, 1.728540000000e02],
                           [0.000000000000e00, 0.000000000000e00, 1.000000000000e00]])
        self.poses = []
        
        try:
            if not all([".png" in x for x in os.listdir(img_file_path)]):
                raise ValueError(
                    "img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError(
                "The designated img_file_path does not exist, please check the path and try again")
        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError(
                "The pose_file_path is not valid or did not lead to a txt file")
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

    def run_optimizer(self, local_window=10):

        """
        Add poses to the optimizer graph
        """
        if len(self.poses) < local_window + 1:
            return

        self.pose_graph = PoseGraphOptimization()
        
        local_poses = self.poses[1:][-local_window:]

        for i in range(1, local_window):
            self.pose_graph.add_vertex(i, local_poses[i])
            self.pose_graph.add_edge((i-1, i), getTransform(local_poses[i], local_poses[i-1]))
            self.pose_graph.optimize(self.args.num_iter)
        
        # self.poses[-local_window+1:] = self.pose_graph.nodes_optimized
        for i in range(local_window - 1):
            self.poses[-local_window + 1 + i] = self.pose_graph.get_pose(i).matrix()
            
    def match(self, kptdes):
        good = []
        if self.matcher_name == 'BF':
            matches = self.matcher.match(
                kptdes['ref']['descriptors'], kptdes['cur']['descriptors'])
            # in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            for i in range(300):
                good.append([matches[i]])

        elif self.matcher_name == 'FLANN' or self.detector_name == 'SIFT':
            matches = self.matcher.knnMatch(
                kptdes['ref']['descriptors'], kptdes['cur']['descriptors'], k=2)
            for m, n in matches:  # Apply ratio test
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            # in the order of their distance.
            good = sorted(good, key=lambda x: x[0].distance)

        kp_ref = np.zeros([len(good), 2])
        kp_cur = np.zeros([len(good), 2])
        match_dist = np.zeros([len(good)])
        for i, m in enumerate(good):
            kp_ref[i, :] = kptdes['ref']['keypoints'][m[0].queryIdx]
            kp_cur[i, :] = kptdes['cur']['keypoints'][m[0].trainIdx]
            match_dist[i] = m[0].distance
        return kp_ref, kp_cur

    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''
        if self.detector_name == 'FAST' and self.matcher_name == 'LK':
            if self.n_features < 2000:
                kps = self.detector.detect(self.old_frame)
                self.p0 = np.array([x.pt for x in kps],
                                   dtype=np.float32).reshape(-1, 1, 2)

            # Calculate optical flow between frames, st holds status of points from frame to frame
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.old_frame, self.current_frame, self.p0, None, **self.lk_params)

            # Save the good points from the optical flow
            self.good_old = self.p0[st == 1]
            self.good_new = self.p1[st == 1]

            # Save the total number of good features
            self.n_features = self.good_new.shape[0]

        else:
            # update keypoints and descriptors
            if self.id == 0:
                self.kptdes['ref'] = self.detect(self.old_frame)
            else:
                self.kptdes['ref'] = self.kptdes['cur']
            self.kptdes['cur'] = self.detect(self.current_frame)

            # match keypoints
            self.good_old, self.good_new = self.match(self.kptdes)

        # compute relative R, t between ref and cur frame
        if self.ransac == 5:
            E, _ = cv2.findEssentialMat(
                self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        elif self.ransac == 8:
            E = self.findEssentialMat()
        _, R, t, _ = cv2.recoverPose(
            E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)

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
                
                self.cur_Rt = convert_to_Rt(self.R, self.t)
                self.poses.append(convert_to_4_by_4(self.cur_Rt))

                if self.args.optimize:
                    print ('optmize!!!!')
                    self.run_optimizer(self.args.local_window)

                    # update cur R, t
                    self.t = self.poses[-1][:3, -1].reshape(-1, 1)


    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)
        return adj_coord.flatten()

    def get_true_coordinates(self):
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
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
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

    def findEssentialMat(self):

        fundamental_matrix, self.good_old, self.good_new = self.RANSAC_8pt(
            self.good_old, self.good_new)
        E = self.K.T @ fundamental_matrix @ self.K
        return E

    def RANSAC_8pt(self, previous_frame_points, current_frame_points):
        """
        Open CV Eight point estimation of fundamental matrix
        :param previous_frame_points_points: Previous frame matched key points
        :param current_frame_points: Current frame matched key points
        :return: Fundamental matrix, inlier points
        """
        _, inliers = cv2.findFundamentalMat(
            previous_frame_points, current_frame_points, cv2.FM_RANSAC)
        previous_frame_inliers = previous_frame_points[inliers.ravel() == 1]
        current_frame_inliers = current_frame_points[inliers.ravel() == 1]
        F, _ = cv2.findFundamentalMat(
            previous_frame_inliers, current_frame_inliers, cv2.FM_8POINT)
        return F.T, previous_frame_inliers, current_frame_inliers

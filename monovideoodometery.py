import numpy as np
import cv2
import os
from optimizer import PoseGraph
from utils import *

class MonoVideoOdometery(object):
    def __init__(self, 
                 args,
                 img_file_path,
                 pose_file_path,
                 detector,
                 focal_length = 718.8560,
                 pp = (607.1928, 185.2157), 
                 lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))):
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

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.args = args
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.id = 0
        self.n_features = 0
        self.poses = []

        try:
            if not all([".png" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        '''Used to determine whether there are remaining frames
           in the folder to process
        
        Returns:
            bool -- Boolean value denoting whether there are still 
            frames in the folder to process
        '''

        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def run_optimizer(self, local_window=10):

        """
        Add poses to the optimizer graph
        """
        if len(self.poses)<local_window+1:
            return

        self.pose_graph = PoseGraph(verbose = True)
        local_poses = self.poses[1:][-local_window:]

        for i in range(1,len(local_poses)):   
            self.pose_graph.add_vertex(i, local_poses[i])
            self.pose_graph.add_edge((i-1, i), getTransform(local_poses[i], local_poses[i-1]))
            self.pose_graph.optimize(self.args.num_iter)
        
        self.poses[-local_window+1:] = self.pose_graph.nodes_optimized

    def visual_odometery(self):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''

        if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)


        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)
        

        # Save the good points from the optical flow
        self.good_old = self.p0[st == 1]
        self.good_new = self.p1[st == 1]


        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, focal=self.focal, pp=self.pp, mask=None)
        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        if self.id < 2:
            self.R = R
            self.t = t
        else:
            absolute_scale = self.get_absolute_scale()
            if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                self.t = self.t + absolute_scale*self.R.dot(t)
                self.R = R.dot(self.R)
            
            self.cur_Rt = convert_to_Rt(self.R, self.t)
            self.poses.append(convert_to_4_by_4(self.cur_Rt))

            if self.args.optimize:
                self.run_optimizer(self.args.local_window)
            
                # update cur R, t
                # self.R = self.poses[-1][:3, :3]
                self.t = self.poses[-1][:3, -1].reshape(-1, 1)
                
        # Save the total number of good features
        self.n_features = self.good_new.shape[0]
        
    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector
        # onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
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


    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''

        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1



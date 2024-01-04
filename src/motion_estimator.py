from abc import ABC, abstractmethod
import cv2


class Estimator(ABC):
    @abstractmethod
    def estimate(self):
        pass


class FivePointsEstimator(Estimator):
    def __init__(self, focal, pp) -> None:
        self.focal = focal
        self.pp = pp

    def estimate(self, good_old, good_new):
        E, _ = cv2.findEssentialMat(
            good_new, good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
        _, R, t, _ = cv2.recoverPose(
            E, good_old, good_new, focal=self.focal, pp=self.pp, mask=None)
        return R, t


class EightPointsEstimator(Estimator):
    def __init__(self, focal, pp, K) -> None:
        self.focal = focal
        self.pp = pp
        self.K = K

    def estimate(self, good_old, good_new):

        fundamental_matrix, self.good_old, self.good_new = self.RANSAC_8pt(
            good_old, good_new)
        E = self.K.T @ fundamental_matrix @ self.K
        _, R, t, _ = cv2.recoverPose(
            E, good_old, good_new, focal=self.focal, pp=self.pp, mask=None)
        return R, t

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

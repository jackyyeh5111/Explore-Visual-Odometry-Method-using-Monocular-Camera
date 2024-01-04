from abc import ABC, abstractmethod
import cv2
from matcher import BFMatcher, FLANNMatcher
from motion_estimator import FivePointsEstimator, EightPointsEstimator


class Factory(ABC):
    @abstractmethod
    def generate(self, args):
        pass


class MotionEstimatorFactory(Factory):
    @staticmethod
    def generate(args):
        if args.motion_estimator == 5:
            estimator = FivePointsEstimator(args.focal, args.pp)
        elif args.motion_estimator == 8:
            estimator = EightPointsEstimator(args.focal, args.pp, args.K)
        return estimator


class FeatureDetectorFactory(Factory):
    @staticmethod
    def generate(args):
        if args.detector_name == 'FAST':
            detector = cv2.FastFeatureDetector_create(
                threshold=25, nonmaxSuppression=True)
        elif args.detector_name == 'BRISK':
            detector = cv2.BRISK_create(thresh=25)
        elif args.detector_name == 'ORB':
            detector = cv2.ORB_create(nfeatures=1000, fastThreshold=20)
        elif args.detector_name == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=2000)
        else:
            raise ValueError('Unknown detector type: {}'.format(detector))

        return detector


class FeatureMatcherFactory(Factory):
    @staticmethod
    def generate(args):
        if args.matcher_name == 'BF':
            matcher = BFMatcher(args.detector_name)
        elif args.matcher_name == 'FLANN':
            matcher = FLANNMatcher(args.detector_name)
        else:
            raise ValueError(
                'Unknown matcher type: {}'.format(args.matcher_name))
        return matcher

from abc import ABC, abstractmethod
import cv2

class Factory(ABC):
    @abstractmethod
    def generate(self):
        pass

class FeatureDetectorFactory(Factory):
    @staticmethod
    def generate(name):
        if name == 'FAST':
            detector = cv2.FastFeatureDetector_create(
                threshold=25, nonmaxSuppression=True)
        elif name == 'BRISK':
            detector = cv2.BRISK_create(thresh=25)
        elif name == 'ORB':
            detector = cv2.ORB_create(nfeatures=1000, fastThreshold=20)
        elif name == 'SIFT':
            detector = cv2.SIFT_create(nfeatures=2000)
        else:
            raise ValueError('Unknown detector type: {}'.format(detector))

        return detector
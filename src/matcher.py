from abc import ABC, abstractmethod
import cv2
import numpy as np


class Matcher(ABC):
    @abstractmethod
    def match(self):
        pass

    def getKeypointPairs(self, kptdes, good):
        kp_ref = np.zeros([len(good), 2])
        kp_cur = np.zeros([len(good), 2])
        match_dist = np.zeros([len(good)])
        for i, m in enumerate(good):
            kp_ref[i, :] = kptdes['ref']['keypoints'][m[0].queryIdx]
            kp_cur[i, :] = kptdes['cur']['keypoints'][m[0].trainIdx]
            match_dist[i] = m[0].distance
        return kp_ref, kp_cur


class BFMatcher(Matcher):
    def __init__(self, detector_name):
        if detector_name == 'SIFT':
            self.matcher = cv2.BFMatcher()
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, kptdes):
        good = []
        matches = self.matcher.match(
            kptdes['ref']['descriptors'], kptdes['cur']['descriptors'])
        # in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        for i in range(300):
            good.append([matches[i]])

        return super().getKeypointPairs(kptdes, good)


class FLANNMatcher(Matcher):
    def __init__(self, detector_name):
        if detector_name == 'SIFT':
            index_params = dict(algorithm=1, trees=5)
        else:
            index_params = dict(algorithm=6, table_number=6,
                                key_size=12, multi_probe_level=2)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, kptdes):
        good = []
        matches = self.matcher.knnMatch(
            kptdes['ref']['descriptors'], kptdes['cur']['descriptors'], k=2)
        for m, n in matches:  # Apply ratio test
            if m.distance < 0.75 * n.distance:
                good.append([m])
        # in the order of their distance.
        good = sorted(good, key=lambda x: x[0].distance)

        return super().getKeypointPairs(kptdes, good)

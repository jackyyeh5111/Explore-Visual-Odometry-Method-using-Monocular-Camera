import os
import cv2
import argparse
import numpy as np
from time import time
from mono_visual_odometry import MonoVisualOdometery
import pathlib
from visualizer import Visualizer

parser = argparse.ArgumentParser()

# different algo choice
parser.add_argument('--detector', '-d', type=str,
                    default='FAST', choices=['FAST', 'BRISK', 'ORB', 'SIFT'])
parser.add_argument('--matcher', '-m', type=str,
                    default='LK', choices=['LK', 'BF', 'FLANN'])
parser.add_argument('--ransac', '-r', type=int, default=5, choices=[5, 8])

# system params
parser.add_argument('--target_num', '-n', type=int, default=-1,
                    help='-1 means that all frames should be processed')
parser.add_argument('--vis', action='store_true',
                    help='visualize intermediate result')

OUTPUT_DIR = 'results'
IMG_PATH = './dataset/sequences/00/image_0/'
POSE_PATH = './dataset/poses/00.txt'
FOCAL = 718.8560  # focal length
PP = (607.1928, 185.2157)  # principle point


def main():
    args = parser.parse_args()
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    log = open('./{}/{}+{}.txt'.format(OUTPUT_DIR,
               args.detector, args.matcher), 'w')
    vo = MonoVisualOdometery(args, IMG_PATH, POSE_PATH, FOCAL, PP)
    visualizer = Visualizer(vo)

    frame_num = 0
    errors = []
    start_time = time()
    while (vo.hasNextFrame()):
        print('========  Processing frame {}  ========'.format(frame_num))
        vo.process_frame()

        # compute error
        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()
        dist = np.linalg.norm(mono_coord - true_coord)
        errors.append(dist)

        # log
        print('pred_x: {}, pred_y: {}, pred_z: {}'.format(
            *[str(round(pt, 2)) for pt in mono_coord]))
        print('true_x: {}, true_y: {}, true_z: {}'.format(
            *[str(round(pt, 2)) for pt in true_coord]))
        print('Pose Error: {}\n'.format(dist))

        log.write(str(frame_num)+' '+str(mono_coord[0])+' '+str(mono_coord[1])+' '+str(mono_coord[2])
                  + ' '+str(true_coord[0])+' '+str(true_coord[1])+' '+str(true_coord[2])+'\n')

        if args.vis:
            visualizer.display()

        frame_num += 1
        if frame_num == args.target_num:
            break

    log.close()

    # output final result
    cv2.imwrite("./{}/{}+{}.png".format(OUTPUT_DIR, args.detector, args.matcher),
                visualizer.get_final_result(np.mean(errors), time() - start_time))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

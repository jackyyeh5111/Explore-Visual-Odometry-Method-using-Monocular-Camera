import os
import cv2
import argparse
import numpy as np
from time import time
from visual_odometry import VisualOdometry
import pathlib
from visualizer import Visualizer

parser = argparse.ArgumentParser()

# different algo choice
parser.add_argument('--detector_name', '-d', type=str, default='FAST',
                    choices=['FAST', 'BRISK', 'ORB', 'SIFT'], help='Feature detector')
parser.add_argument('--matcher_name', '-m', type=str, choices=[
                    'BF', 'FLANN'], help='Feature matcher. Only useful when flag use_tracking is false')
parser.add_argument('--motion_estimator', '--mo', type=int, default=5, choices=[
                    5, 8], help='5 stands for Nister\'s 5-point algo, 8 stands for 8-point algo. Both algo includes RANSAC outlier filter.')
parser.add_argument('--use_tracking', action='store_true',
                    help='Use Lucas–Kanade algo to tack features')
parser.add_argument('--focal', type=float, default=718.8560)
parser.add_argument('--pp', type=tuple, default=(607.1928,
                    185.2157), help='principal points')

# system params
parser.add_argument('--target_num', '-n', type=int, default=-1,
                    help='-1 means that all frames should be processed')
parser.add_argument('--vis', action='store_true',
                    help='visualize intermediate result')

OUTPUT_DIR = 'results'
IMG_PATH = './dataset/sequences/00/image_0/'
POSE_PATH = './dataset/poses/00.txt'


def main():
    args = parser.parse_args()

    if not args.use_tracking and not args.matcher_name:
        parser.error(
            "--matcher_name is required when use feature matching pipeline.")

    # sanity check
    try:
        if not all([".png" in x for x in os.listdir(IMG_PATH)]):
            raise ValueError(
                "img_file_path is not correct and does not exclusively png files")
    except Exception as e:
        print(e)
        raise ValueError(
            "The designated img_file_path does not exist, please check the path and try again")

    if args.detector_name == 'FAST' and not args.use_tracking:
        raise ValueError(
            'OpenCV supports FAST detector only, not including FAST descriptor. Therefore FAST detector is only used in tracking pipeline in this project')

    # load gt path
    try:
        with open(POSE_PATH) as f:
            gt_pose = f.readlines()
    except Exception as e:
        print(e)
        raise ValueError(
            "The pose_file_path is not valid or did not lead to a txt file")

    # initialize
    log = open('./{}/{}+{}.txt'.format(OUTPUT_DIR,
               args.detector_name, args.matcher_name), 'w')
    vo = VisualOdometry(args, gt_pose, IMG_PATH)
    visualizer = Visualizer(vo)

    # process sequential frames
    frame_num = 0
    errors = []
    start_time = time()
    while (vo.hasNextFrame()):
        print('========  Processing frame {}  ========'.format(frame_num))
        vo.process_frame()

        # compute error
        pred_coord = vo.pred_coordinates
        gt_coord = vo.gt_coordinates

        dist = np.linalg.norm(pred_coord - gt_coord)
        errors.append(dist)

        # log
        print('pred_x: {}, pred_y: {}, pred_z: {}'.format(
            *[str(round(pt, 2)) for pt in pred_coord]))
        print('gt_x: {}, gt_y: {}, gt_z: {}'.format(
            *[str(round(pt, 2)) for pt in gt_coord]))
        print('Pose Error: {}\n'.format(dist))

        log.write(str(frame_num)+' '+str(pred_coord[0])+' '+str(pred_coord[1])+' '+str(pred_coord[2])
                  + ' '+str(gt_coord[0])+' '+str(gt_coord[1])+' '+str(gt_coord[2])+'\n')

        if args.vis:
            visualizer.display()

        frame_num += 1
        if frame_num == args.target_num:
            break

    log.close()
    
    # output final result
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    vis_result = visualizer.get_final_result(np.mean(errors), time() - start_time)
    output_path = "./{}/{}+{}.png".format(OUTPUT_DIR, args.detector_name, args.matcher_name)
    print('output_path:', output_path)
    cv2.imwrite(output_path, vis_result)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

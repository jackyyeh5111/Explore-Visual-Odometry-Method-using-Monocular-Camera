import numpy as np
import cv2
from monovideoodometery import MonoVideoOdometery
import os
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--feature', '-f', type=str, choices=['sift', 'fast', 'orb'])
parser.add_argument('--target_num', '-n', type=int, default=-1, help='-1 means that all frames should be processed')
parser.add_argument('--vis', action='store_true', help='visualize intermediate result')
parser.add_argument('--optimize', action='store_true', help='enable pose graph optimization')
parser.add_argument('--local_window', default=5, type=int, help='number of frames to run the optimization')
parser.add_argument('--num_iter', default=100, type=int, help='number of max iterations to run the optimization')

OUTPUT_DIR = 'output'

def main():
    args = parser.parse_args()

    print('======= Initial arguments =======')
    for key, val in vars(args).items():
        print("name: {} => {}".format(key, val))

    img_path = './dataset/sequences/00/image_0/'
    pose_path = './dataset/poses/00.txt'
    
    focal = 718.8560
    pp = (607.1928, 185.2157)
    R_total = np.zeros((3, 3))
    t_total = np.empty(shape=(3, 1))

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


    # Create some random colors
    color = np.random.randint(0, 255, (5000, 3))

    detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    vo = MonoVideoOdometery(args, img_path, pose_path, detector, focal, pp, lk_params)
    traj = np.zeros(shape=(600, 800, 3))

    # mask = np.zeros_like(vo.current_frame)
    # flag = False
    cnt = 0
    rmse_errors = []
    while (vo.hasNextFrame()):

        cnt += 1
        if cnt == args.target_num + 1:
            break

        print("\n----- Processing frames {} -----".format(cnt))
        frame = vo.current_frame

        vo.process_frame()

        print(vo.get_mono_coordinates())

        mono_coord = vo.get_mono_coordinates()
        true_coord = vo.get_true_coordinates()

        rmse_error = np.linalg.norm(mono_coord - true_coord)
        rmse_errors.append(rmse_error)
        print("MSE Error: ", rmse_error)
        print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
        print("true_x: {}, true_y: {}, true_z: {}".format(
            *[str(pt) for pt in true_coord]))

        draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
        true_x, true_y, true_z = [int(round(x)) for x in true_coord]

        traj = cv2.circle(traj, (true_x + 400, true_z + 100),
                        1, list((0, 0, 255)), 4)
        traj = cv2.circle(traj, (draw_x + 400, draw_z + 100),
                        1, list((0, 255, 0)), 4)

        cv2.putText(traj, 'Actual Position:', (140, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(traj, 'Red', (270, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(traj, 'Estimated Odometry Position:', (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(traj, 'Green', (270, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if args.vis:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

            cv2.imshow('trajectory', traj)

        if cnt % 100 == 0:
            output_path = os.path.join(
                OUTPUT_DIR, 'trajectory_{}.png'.format(cnt))
            cv2.imwrite(output_path, traj)

    mean_rmse_error = np.mean(rmse_errors)
    print('avg rmse_error:', mean_rmse_error)
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # display avg rmse error
    # Put the text on the image
    cv2.putText(traj, 'avg rmse_error: {:.3f}'.format(mean_rmse_error), (140, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    output_path = os.path.join(OUTPUT_DIR, 'trajectory_w{}_i{}.png'.format(
        args.local_window, args.num_iter))
    cv2.imwrite(output_path, traj)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import os
import cv2
import argparse
import numpy as np
from time import time
from mono_visual_odometry import MonoVisualOdometery
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--detector', '-d', type=str,
                    default='FAST', choices=['FAST', 'BRISK', 'ORB', 'SIFT'])
parser.add_argument('--matcher', '-m', type=str,
                    default='LK', choices=['LK', 'BF', 'FLANN'])
parser.add_argument('--target_num', '-n', type=int, default=-1,
                    help='-1 means that all frames should be processed')
parser.add_argument('--ransac', '-r', type=int, default=5, choices=[5, 8])
parser.add_argument('--vis', action='store_true',
                    help='visualize intermediate result')
parser.add_argument('--optimize', action='store_true',
                    help='enable pose graph optimization')
parser.add_argument('--local_window', default=5, type=int,
                    help='number of frames to run the optimization')
parser.add_argument('--num_iter', default=100, type=int,
                    help='number of max iterations to run the optimization')

OUTPUT_DIR = 'results'

def main():
    args = parser.parse_args()
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    log = open('./{}/{}+{}.txt'.format(OUTPUT_DIR, args.detector, args.matcher), 'w')
    img_path = './dataset/sequences/00/image_0/'
    pose_path = './dataset/poses/00.txt'
    focal = 718.8560
    pp = (607.1928, 185.2157)
    vo = MonoVisualOdometery(args,
                             img_path, 
                             pose_path, 
                             args.detector, 
                             args.matcher, 
                             focal, 
                             pp, 
                             args.ransac)
    traj = np.zeros(shape=(600, 800, 3))

    f_num = 0
    f_tot = len(os.listdir(img_path)) if args.target_num == - \
        1 else args.target_num
    errors = []
    flag = False
    try:
        start_time = time()
        while (vo.hasNextFrame()):
            frame = vo.current_frame
            
            if args.vis:
                cv2.imshow('frame', frame)
                k = cv2.waitKey(1)

                if k == 27:
                    break
                if k == 121:
                    flag = not flag
                    def toggle_out(flag): return "On" if flag else "Off"
                    print("Flow lines turned ", toggle_out(flag))

            vo.process_frame()
            mono_coord = vo.get_mono_coordinates()
            true_coord = vo.get_true_coordinates()
            dist = np.linalg.norm(mono_coord - true_coord)
            errors.append(dist)

            print('========  Processing frame {}  ========'.format(f_num))
            print('pred_x: {}, pred_y: {}, pred_z: {}'.format(
                *[str(round(pt, 2)) for pt in mono_coord]))
            print('true_x: {}, true_y: {}, true_z: {}'.format(
                *[str(round(pt, 2)) for pt in true_coord]))
            print('Pose Error: {}\n'.format(dist))

            log.write(str(f_num)+' '+str(mono_coord[0])+' '+str(mono_coord[1])+' '+str(mono_coord[2])
                      + ' '+str(true_coord[0])+' '+str(true_coord[1])+' '+str(true_coord[2])+'\n')

            draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
            true_x, true_y, true_z = [int(round(x)) for x in true_coord]
            traj = cv2.circle(traj, (true_x+400, true_z+100),
                              1, list((0, 0, 255)), 4)
            traj = cv2.circle(traj, (draw_x+400, draw_z+100),
                              1, list((0, 255, 0)), 4)

            cv2.putText(traj, 'Ground True Trajectory:', (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(traj, 'Red', (260, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(traj, 'Estimated Trajectory:', (60, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(traj, 'Green', (240, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if args.vis:
                cv2.imshow('trajectory', traj)

            f_num = f_num + 1
            if f_num >= f_tot:
                break

        log.close()
        end_time = time()
        cv2.putText(traj, 'Average RMSE Error: {:.3f} m'.format(np.mean(errors)), (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(traj, 'Elapsed Time: {:.3f} s'.format(end_time - start_time), (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(
            "./{}/{}+{}.png".format(OUTPUT_DIR, args.detector, args.matcher), traj)
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        log.close()
        cv2.imwrite(
            "./{}/{}+{}.png".format(OUTPUT_DIR, args.detector, args.matcher), traj)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

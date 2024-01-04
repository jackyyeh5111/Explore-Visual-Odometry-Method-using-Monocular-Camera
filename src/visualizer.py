import cv2
import numpy as np

CANVAS_SHAPE = (600, 800, 3)


class Visualizer:
    def __init__(self, vo) -> None:
        self.vo = vo
        self.traj = np.zeros(shape=CANVAS_SHAPE)

    def display(self):
        cv2.imshow('frame', self.vo.current_frame)
        cv2.waitKey(1)

        draw_x, draw_y, draw_z = [int(round(x))
                                  for x in self.vo.pred_coordinates]
        true_x, true_y, true_z = [int(round(x))
                                  for x in self.vo.gt_coordinates]
        self.traj = cv2.circle(self.traj, (true_x+400, true_z+100),
                               1, list((0, 0, 255)), 4)
        self.traj = cv2.circle(self.traj, (draw_x+400, draw_z+100),
                               1, list((0, 255, 0)), 4)

        cv2.putText(self.traj, 'Ground True Trajectory:', (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.traj, 'Red', (260, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.traj, 'Estimated Trajectory:', (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.traj, 'Green', (240, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('trajectory', self.traj)

    def get_final_result(self, error_mean, elasped_time):
        cv2.putText(self.traj, 'Average RMSE Error: {:.3f} m'.format(error_mean), (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.traj, 'Elapsed Time: {:.3f} s'.format(elasped_time), (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return self.traj
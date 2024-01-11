# Monocular Feature-Based Visual Odometry using Python OpenCV

Welcome! This is our final project for course CS543-Computer-Vision in 2023 Fall. The course page can be found [here](http://luthuli.cs.uiuc.edu/~daf/courses/CV23/CV23.html).

The project implements a 2D-to-2D feature-based visual odometry, conducting experiments on different various methods across different components of the visual odometry process.

Please check out my [portfolio post](https://jackyyeh5111.github.io/monocular-feature-based-visual-odometry/) for a greater detailed description.

## Overview
![ezgif-5-c1d071cdc8](https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/f0ebf1fc-1dca-4808-ada1-c94ee6ac4ef8)

## Method
<img src='https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/dd39ccbc-a4a0-4e82-9673-2e1d4dd243ad' width='500'>

The project includes two typical pipelines of feature-based visual odometry: matching & tracking. In general, the system receives frames in a sequential manner. Then it identifies corresponding feature pairs between $t-1$ and $t$, and finally do motion estimation and optimization across frames.

We implement various algorithms across each components and compare their performance:
1. Feature detection: ORB, SIFT, BRIEF, FAST
2. Feature matching: BF, FLANN
3. Feature tracking: Lucas–Kanade
3. Motion estimation: Nister’s 5-point and 8-point algorithm with RANSAC.
4. (TODO) Optimization: pose-graph

## Folder Structure
```
src
├── factories.py
├── main.py
├── matcher.py
├── motion_estimator.py
├── utils.py
├── visual_odometry.py
└── visualizer.py
```

- `factories.py`: implements Factory design pattern for creating objects without specifying their concrete classes.
- `main.py`: main entry point.
- `matcher.py` implements feature matcher objects.
- `motion_estimator.py` implements motion estimator objects.
- `utils.py` includes utility functions.
- `visual_odometry.py` performs visual odometery algorithm.
- `visualizer.py` performs all visualization tasks.

## Quick Starter Guide
### Installation
1. Clone repo
    ```
    $ git clone https://github.com/jackyyeh5111/Explore-Visual-Odometry-Method-using-Monocular-Camera.git
    $ cd Explore-Visual-Odometry-Method-using-Monocular-Camera
    ```
2. Activate virtualenv and install dependencies
    ```
    $ pip install -r requirements.txt
    ```

### Usage
```bash
$ python3 src/main.py [params...]

# ex:
# python src/main.py --vis --use_tracking
# python src/main.py --vis -d SIFT -m BF --mo 5 
```
important params:
- `-d`: Feature detector name (choices=['FAST', 'BRISK', 'ORB', 'SIFT'])
- `-m`: Feature matcher. Only useful when flag use_tracking is false (choices=['BF', 'FLANN'])
- `--mo`: Motion estimator. (choices=['5', '8']) 5 stands for Nister\'s 5-point algo, 8 stands for 8-point algo. Both algo includes RANSAC outlier filter.
- `-n`: Number of frames should be processed.
- `--use_tracking`: Use Lucas–Kanade (optical flow) algo to track features. Otherwise, use feature matcher to match features across different frames.
- `--vis`: visualize intermediate result.

## Dataset
Kitti sequence 00 ([link](https://www.cvlibs.net/datasets/kitti/eval_odometry.php))

## Results
Trajectory estimation using different combinations of feature detectors and matchers:
<img width="750" alt="image" src="https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/ad78a1aa-1d99-4973-8455-fb50673d327d">

Comparison of trajectory estimation results between (a) feature tracking and (b) feature matching:
<img width="500" alt="image" src="https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/521612c5-9075-44f5-a84a-2fd008680864">


Trajectory estimation using (a) 5-point RANSAC. (b) 8-point RANSAC:
<img width="500" alt="image" src="https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/09203608-05cc-4bfe-bfac-f37e997252c7">

## TODO
- [ ] Integrate pose-graph optimization
- [ ] Extract more params in config file(some params are hard-coded in code now)
- [ ] Implement 3D-2D visual odometry

## Reference
- [Great blog post by Avi Singh](https://avisingh599.github.io/vision/monocular-vo/)
- [Monocular Video Odometry Using OpenCV repository by Ali Shobeiri](https://github.com/alishobeiri/Monocular-Video-Odometery)

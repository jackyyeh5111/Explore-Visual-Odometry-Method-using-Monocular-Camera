# Monocular Feature-Based Visual Odometry using Python OpenCV

Welcome! This is our final project for course CS543-Computer-Vision in 2023 Fall. The course page can be found [here](http://luthuli.cs.uiuc.edu/~daf/courses/CV23/CV23.html).

The project implements a 2D-to-2D feature-based visual odometry, conducting experiments on different various methods across different components of the visual odometry process.

Please check out my portfolio post for a greater detailed description.

## Overview

## Method
<img src='pics/pipeline.png' width='500'>

The project explores two typical pipelines of feature-based visual odometry: matching & tracking. In general, the system receives frames in a sequential manner. Then it identifies corresponding feature pairs between $t-1$ and $t$

built vision-based lane following system from scratch. Lane detector identifies the lane from the captured frame and provides imaginary waypoints candidates for the controller. Next, the controller selects the best waypoint based on the vehicle state, and sends out next control signal.

The whole system is integrated with ROS. It consists of four primary components:
1. Camera calibration
2. Lane detection
3. State estimation
4. Controller


## Quick Starter Guide

## Results

## TODO

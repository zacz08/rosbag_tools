#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# for .bag writing 
import rosbag
# to store the IMU/INS measurements as a ROS message
from sensor_msgs.msg import Imu
# for image interpretation and manipulation
import cv2
# to convert from a cv2 image to a ROS compatible image data
from cv_bridge import CvBridge
# to read .hdf5 sensor records files
import h5py    
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import re
import shlex
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--outputFile", help="output file", required=True)
parser.add_argument("-l", "--leftImgDir", help="directory of left images (can be the same as rightImgDir)", required=True)
parser.add_argument("-r", "--rightImgDir", help="enable stereo processing, directory of right images (can be the same as leftImgDir)")
parser.add_argument("-i", "--imuFile", help="CSV.txt of imu data")

args = parser.parse_args()

try:
    if args.help:
        parser.print_help()
        exit()
except AttributeError:
    pass

outputPath = os.path.abspath(f"{args.outputFile}")
assert outputPath[-4:] == ".bag"  # validate output file

leftImagePath = os.path.abspath(f"{args.leftImgDir}")

if args.rightImgDir:
    rightImagePath = os.path.abspath(f"{args.rightImgDir}")

if args.imuFile:
    imuPath = os.path.abspath(f"{args.imuFile}")
else:
    imuPath = None

if re.findall("kitti", leftImagePath):
    from bag_kitti import KittiBagGen
    kbg = KittiBagGen(imgDir=leftImagePath, imuDir=imuPath, output_filename=outputPath)
    exit()

if re.findall("malaga", leftImagePath):
    # list images in left img dir
    leftImageList = sorted(os.listdir(leftImagePath))

    # lists for the file path and the timestamp of each left camera image
    leftCameraPaths = []
    leftCameraTimestamps = []

    # for each of the stero files
    for img in leftImageList:
        if re.search("(?<=CAM)[^\s]*(?<=left)[^\s]*.jpg", img):  # match "CAM~~~left~~~.jpg"
            # store all files that are from the left camera
            leftCameraPaths.append(leftImagePath + "/" + img)

            # and separately store all of their timestamps
            timestamp = re.search("(?<=_)\d+.(?!j)\d*", img)  # extract XXX(.XXX) from filename
            leftCameraTimestamps.append(float(timestamp.group(0)))

    # lists for the file path and the timestamp of each right camera image
    rightCameraPaths = []
    rightCameraTimestamps = []

    if args.rightImgDir:
        # list images in right img dir
        rightImageList = sorted(os.listdir(rightImagePath))

        # for each of the stero files
        for img in rightImageList:
            if re.search("(?<=CAM)[^\s]*(?<=right)[^\s]*.jpg", img):  # match "CAM~~~right~~~.jpg"
                # store all files that are from the right camera
                rightCameraPaths.append(rightImagePath + "/" + img)

                # and separately store all of their timestamps
                timestamp = re.search("(?<=_)\d+.(?!j)\d*", img)  # extract XXX(.XXX) from filename
                rightCameraTimestamps.append(float(timestamp.group(0)))

    # image dimensions
    imWidth = 1024
    imHeight = 768

    if args.imuFile:
        # read the collected imu data into a list
        with open(imuPath) as f1:
            imuDataFile = f1.readlines()

        # split the list items by spaces
        imuData = []
        for line in imuDataFile:
            imuData.append(shlex.split(line))

        # get particular readings of interest
        imuDataTimestamps = [float(row[0]) for row in imuData[1:]]
        imuDataAccX = [float(row[1]) for row in imuData[1:]]
        imuDataAccY = [float(row[2]) for row in imuData[1:]]
        imuDataAccZ = [float(row[3]) for row in imuData[1:]]
        imuDataGyrX = [float(row[6]) for row in imuData[1:]]    # x is forward (roll)
        imuDataGyrY = [float(row[5]) for row in imuData[1:]]    # y is left (pitch)
        imuDataGyrZ = [float(row[4]) for row in imuData[1:]]    # z is upward (yaw)

        # concatenate each set of x,y,z readings into rows
        accelerometerData = [[imuDataAccX[i], imuDataAccY[i], imuDataAccZ[i]] for i in range(len(imuDataTimestamps))]
        gyroscopeData = [[imuDataGyrX[i], imuDataGyrY[i], imuDataGyrZ[i]] for i in range(len(imuDataTimestamps))]

    seq = 0 # each image should be given an index in the sequence

    # open our .bag file to write to
    bag = rosbag.Bag(f"{outputPath}", "w")

    for imagePath, imageTimestamp in zip(leftCameraPaths, leftCameraTimestamps):
        # ---------- IMAGE RECORD ---------- #
        # if the image exists and isn't a dud
        if (os.path.isfile(imagePath)) and (os.stat(imagePath).st_size != 0):

            print(f"Adding {seq}: {imagePath[imagePath.rindex('/') + 1:]}")

            # read the image
            image = cv2.imread(imagePath)

            # convert from cv2 to ROS by creating an Image(). This auto allocates
                # the image data to the .data field so headers/config can be
                # added below
            bridge = CvBridge()
            #img_msg.data = image
            img_msg = bridge.cv2_to_imgmsg(image, "passthrough")

            # img_msg format:
            # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html

            # get the timestamp of this frame
            timestamp = imageTimestamp
            # the seconds part of the timestamp is just the whole number
            timestampSec = math.floor(timestamp)
            # the nanoseconds part of the timestamp is then the decimal part
            timestampnSec = int((timestamp - timestampSec) * pow(10, 9))

            # set image timestamp
            img_msg.header.stamp.secs = timestampSec
            img_msg.header.stamp.nsecs = timestampnSec
            # number in sequence
            img_msg.header.seq = seq
            # frame source
            img_msg.header.frame_id = "cam0"
            # dimensions
            img_msg.width = imWidth
            img_msg.height = imHeight
            img_msg.step = imWidth * 3  # (image width in bytes)
            # encoding
            img_msg.encoding = "bgr8"

            # write image to the bag file under the 'camera/left/image_raw' topic
            bag.write("/cam0/image_raw", img_msg, img_msg.header.stamp)

            # increment seq num
            seq = seq + 1

    if args.rightImgDir:
        # reset sequence ID for Right camera recording
        seq = 0

        for imagePath, imageTimestamp in zip(rightCameraPaths, rightCameraTimestamps):
            # ---------- IMAGE RECORD ---------- #
            # if the image exists and isn't a dud
            if (os.path.isfile(imagePath)) and (os.stat(imagePath).st_size != 0):

                print(f"Adding {seq}: {imagePath[imagePath.rindex('/') + 1:]}")

                # read the image
                image = cv2.imread(imagePath)

                # convert from cv2 to ROS by creating an Image(). This auto allocates
                    # the image data to the .data field so headers/config can be
                    # added below
                bridge = CvBridge()
                #img_msg.data = image
                img_msg = bridge.cv2_to_imgmsg(image, "passthrough")

                # img_msg format:
                # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html

                # get the timestamp of this frame
                timestamp = imageTimestamp
                # the seconds part of the timestamp is just the whole number
                timestampSec = math.floor(timestamp)
                # the nanoseconds part of the timestamp is then the decimal part
                timestampnSec = int((timestamp - timestampSec) * pow(10, 9))

                # set image timestamp
                img_msg.header.stamp.secs = timestampSec
                img_msg.header.stamp.nsecs = timestampnSec
                # number in sequence
                img_msg.header.seq = seq
                # frame source
                img_msg.header.frame_id = "cam0"
                # dimensions
                img_msg.width = imWidth
                img_msg.height = imHeight
                img_msg.step = imWidth * 3  # (image width in bytes)
                # encoding
                img_msg.encoding = "bgr8"

                # write image to the bag file under the 'cam0/image_raw' topic
                bag.write("/cam1/image_raw", img_msg, img_msg.header.stamp)

                # increment seq num
                seq = seq + 1

    if args.imuFile:
        # reset sequence ID for IMU recording
        seq = 0

        for measurementTimestamp, accelLine, gyroLine in zip(imuDataTimestamps, accelerometerData, gyroscopeData):

            print(f"Adding {seq}: IMU, t:{measurementTimestamp}")

            # get the timestamp of this frame
            timestamp = measurementTimestamp
            # the seconds part of the timestamp is just the whole number
            timestampSec = math.floor(timestamp)
            # the nanoseconds part of the timestamp is then the decimal part
            timestampnSec = int((timestamp - timestampSec) * pow(10, 9))

            # create an imu_msg for our inertial data
            imu_msg = Imu()

            # imu_msg format:
            # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html

            # ---------- IMU RECORD ---------- #
            # header info
            imu_msg.header.frame_id = "/imu0"
            imu_msg.header.seq = seq
            imu_msg.header.stamp.secs = timestampSec
            # microseconds to nanoseconds again
            imu_msg.header.stamp.nsecs = timestampnSec

            # linear accelerations (m/s^2)
            imu_msg.linear_acceleration.x = accelLine[0]
            imu_msg.linear_acceleration.y = accelLine[1]
            imu_msg.linear_acceleration.z = accelLine[2]

            # angular rates (rad/s)
            imu_msg.angular_velocity.x = gyroLine[0]
            imu_msg.angular_velocity.y = gyroLine[1]
            imu_msg.angular_velocity.z = gyroLine[2]

            # TODO: attitude

            # get the roll/pitch/yaw values (radians)
            #roll = float(splitLine[17])
            #pitch = float(splitLine[18])
            #yaw = float(splitLine[19])

            # generate quaternions from euler angles & assign
            #qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            #qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            #qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            #qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            #imu_msg.orientation.x = attitudeLine[1]
            #imu_msg.orientation.y = attitudeLine[2]
            #imu_msg.orientation.z = attitudeLine[3]
            #imu_msg.orientation.w = attitudeLine[0]

            imu_msg.orientation_covariance = [-1 for i in imu_msg.orientation_covariance]

            # write the imu_msg to the bag file
            bag.write("imu0", imu_msg, imu_msg.header.stamp)

            # increment the sequence counter
            seq = seq + 1

    # close the .bag file
    print("Closing bag")
    bag.close()

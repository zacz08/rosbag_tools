# to store the IMU/INS measurements as a ROS message
# for image interpretation and manipulation
# to read .hdf5 sensor records files
import datetime as dt
import os

import cv2
import numpy as np
import pandas as pd
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu


class KittiBagGen:
    def __init__(self, imgDir=None, imuDir=None, output_filename=None):
        """
        # :param h5_filename: filename to store dataframe as hdf5
        :param imgDir: path to left image directory for sequence
        :param imuDir: path to imu data directory for sequence
        """
        # if not h5_filename:
        #     raise ValueError("Storage filename not provided")
        # self.h5_filename = str(h5_filename)
        if not output_filename:
            self.of = "kitti_latest.bag"
        else:
            self.of = output_filename

        if not os.path.isdir(imgDir):
            raise NotADirectoryError
            # check dir contents?

        if imuDir and not os.path.isdir(imuDir):
            raise NotADirectoryError
            # check dir contents?

        self.imgDir = imgDir
        self.imuDir = imuDir

        # self.store = pd.HDFStore('dataset.h5')
        # self.data = pd.DataFrame()
        # self.store["data"] = self.data

        self.write_bag()

    def write_bag(self):
        img_names = os.listdir(f"{self.imgDir}/image_0")
        img_names.sort()
        times = []
        with open(f"{self.imgDir}/times.txt") as f:
            times = f.readlines()

        if not times:
            raise ValueError('no timestamps extracted')

        time_offset = 0.01
        times = [float(i) + time_offset for i in times]

        if len(times) != len(img_names):
            raise ValueError('Incompatible timestamp/images list')

        # imgs = []
        # for n, name in enumerate(img_names):
        #     if n % int(len(img_names) / 10) == 0:
        #         print(f"{n} / {len(img_names)}")
        #     imgs.append(cv2.imread(f"{self.imgDir}/image_0/{name}"))
        # self.data.assign(image_0=imgs)
        # del imgs

        with rosbag.Bag(self.of, "w") as bag:

            for i, (t, img_name) in enumerate(zip(times, img_names)):
                if i % int(len(img_names) / 20) == 0:
                    print(f"{2 * i:5d} / {2 * len(img_names):5d} images bagged")
                image = (
                cv2.imread(f"{self.imgDir}/image_0/{img_name}"), cv2.imread(f"{self.imgDir}/image_1/{img_name}"))
                # convert from cv2 to ROS
                bridge = (CvBridge(), CvBridge())
                img_msg = (
                bridge[0].cv2_to_imgmsg(image[0], "passthrough"), bridge[1].cv2_to_imgmsg(image[1], "passthrough"))

                # img_msg format:
                # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html

                # left image
                img_msg[0].header.stamp.secs = int(np.floor(t))
                img_msg[0].header.stamp.nsecs = int(np.round((t - np.floor(t)) * 10 ** 9, 0))
                img_msg[0].header.seq = i
                # frame source
                img_msg[0].header.frame_id = "cam0"
                # definitions for byte handling
                img_msg[0].height = image[0].shape[0]
                img_msg[0].width = image[0].shape[1]
                img_msg[0].step = image[0].shape[1] * image[0].shape[2]  # (image width in bytes)
                # encoding
                img_msg[0].encoding = "bgr8"

                # right image
                img_msg[1].header.stamp.secs = int(np.floor(t))
                img_msg[1].header.stamp.nsecs = int(np.round((t - np.floor(t)) * 10 ** 9, 0))
                img_msg[1].header.seq = i
                # frame source
                img_msg[1].header.frame_id = "cam0"
                # definitions for byte handling
                img_msg[1].height = image[1].shape[0]
                img_msg[1].width = image[1].shape[1]
                img_msg[1].step = image[1].shape[1] * image[1].shape[2]  # (image width in bytes)
                # encoding
                img_msg[1].encoding = "bgr8"

                # write image to the bag file under the 'camera/left/image_raw' topic
                bag.write("/cam0/image_raw", img_msg[0], img_msg[0].header.stamp)
                bag.write("/cam1/image_raw", img_msg[1], img_msg[1].header.stamp)

            if self.imuDir:
                imus = os.listdir(f"{self.imuDir}/data")
                imus.sort()
                times = []
                with open(f"{self.imuDir}/times.txt", "r") as f:
                    init_time = None
                    zero_timedelta = dt.datetime(1900, 1, 1)
                    for time in f.readlines():
                        time, nanoseconds = time.strip().split('.')
                        nanoseconds = float(nanoseconds) / (10 ** 9)
                        timedt = dt.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                        time = (timedt - zero_timedelta).total_seconds() + nanoseconds
                        if not init_time:
                            init_time = time
                        time = time - init_time + time_offset
                        times.append(time)

                imu_keys = ["timestamp"]
                with open(f"{self.imuDir}/dataformat.txt", "r") as f:
                    line = f.readline()
                    while line:
                        imu_keys.append(line.strip().split(":")[0])
                        line = f.readline()

                imu_data = []
                for filename, timestamp in zip(imus, times):
                    with open(f"{self.imuDir}/data/{filename}") as f:
                        vals = f.readline().strip().split(' ')
                        imu_data.append([timestamp, *[float(i) for i in vals]])
                data = pd.DataFrame(imu_data, columns=imu_keys)

                for i in range(data.shape[0]):
                    repeat_count = 2

                    if i % int(data.shape[0] / 10) == 0:
                        print(f"{i * repeat_count:5d} / {data.shape[0] * repeat_count:5d} imu points bagged")

                    for j in range(repeat_count):
                        imu_msg = Imu()

                        # imu_msg format:
                        # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html

                        # ---------- IMU RECORD ---------- #
                        # header info
                        imu_msg.header.frame_id = "/imu0"
                        imu_msg.header.seq = i
                        t = data.loc[i]["timestamp"] + 0.01 * j
                        imu_msg.header.stamp.secs = int(np.floor(t))
                        imu_msg.header.stamp.nsecs = int(np.round((t - np.floor(t)) * 10 ** 9, 0))

                        imu_msg.orientation.w = 1.0
                        imu_msg.orientation_covariance = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

                        # linear accelerations (m/s^2)
                        imu_msg.linear_acceleration.x = data.loc[i]["ax"] / float(repeat_count)
                        imu_msg.linear_acceleration.y = data.loc[i]["ay"] / float(repeat_count)
                        imu_msg.linear_acceleration.z = data.loc[i]["az"] / float(repeat_count)

                        # angular rates (rad/s)
                        imu_msg.angular_velocity.x = data.loc[i]["wx"] / float(repeat_count)
                        imu_msg.angular_velocity.y = data.loc[i]["wy"] / float(repeat_count)
                        imu_msg.angular_velocity.z = data.loc[i]["wz"] / float(repeat_count)

                        # imu_msg.orientation_covariance = [-1 for i in imu_msg.orientation_covariance]

                        # write the imu_msg to the bag file
                        bag.write("imu0", imu_msg, imu_msg.header.stamp)

        print("Bag closed, process complete")


if __name__ == "__main__":
    for seq in range(0, 1):
        kg = KittiBagGen(imgDir=f"/home/dom-ubuntu/Documents/fyp/datasets/kitti/data_odometry_gray/dataset/sequences/{seq:02d}",
                         imuDir=f"/home/dom-ubuntu/Documents/fyp/datasets/kitti/data_odometry_imu/dataset/sequences/{seq:02d}",
                         output_filename=f"/home/dom-ubuntu/Documents/kitti_{seq:02d}.bag")

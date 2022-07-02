import re
import sys

import cv2
import numpy as np

import rosbag
# from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

import image_distort

import multiprocessing

# for file path interpretation and representation    
import os

# input_bag = 'euroc_MH_04.bag'
# output_bag = 'euroc_MH_04_alt004.bag'


class ReBagger:
    def __init__(self, base_bagname, output_bagname):
        self.input_bagname = base_bagname
        self.output_bagname = output_bagname

    def __enter__(self):
        self.input_bag = rosbag.Bag(self.input_bagname, 'r')
        self.output_bag = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.input_bag.close()
        try:
            self.output_bag.close()
        except AttributeError:
            pass
        cv2.destroyAllWindows()

    def cv2_to_imgmsg(self, cv_image, encoding):
        # cv_bridge is conventional way to convert ros -> img, but throws import error despite being present on the system
        # https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height  # integer division
        return img_msg

    def imgmsg_to_cv2(self, img_msg):
        # cv_bridge is conventional way to convert ros -> img, but throws import error despite being present on the system
        # https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
        if img_msg.encoding == "bgr8":
            dtype = np.dtype("uint8")  # Hardcode to 8 bits...
            dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
            image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                                      # and three channels of data. Since OpenCV works with bgr natively, we don't need to
                                      # reorder the channels.
                                      dtype=dtype, buffer=img_msg.data)
            # If the byt order is different between the message and the system.
            if img_msg.is_bigendian == (sys.byteorder == 'little'):
                image_opencv = image_opencv.byteswap().newbyteorder()
            return image_opencv

        if img_msg.encoding == "mono8":
            dtype = np.dtype("uint8")  # Hardcode to 8 bits...
            dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
            image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width),
                                      # and three channels of data. Since OpenCV works with bgr natively, we don't need to
                                      # reorder the channels.
                                      dtype=dtype, buffer=img_msg.data)
            # If the byt order is different between the message and the system.
            if img_msg.is_bigendian == (sys.byteorder == 'little'):
                image_opencv = image_opencv.byteswap().newbyteorder()
            return image_opencv

        print(f"Support for images of type: {img_msg.encoding}, is not yet implemented.")

    def get_topic_details(self, bag, print_output=True):
        info = {topic: details.msg_type for topic, details in bag.get_type_and_topic_info().topics.items()}
        if print_output:
            print(*[f"{key:16s}\t-\t{value:20s}" for key, value in info.items()], sep="\n")  # display
        print("test get_topic_details*******")

    def generate_bag(self, blur=None, occ=None, res=None, disp=None):
        with rosbag.Bag(self.output_bagname, 'w') as self.output_bag:

            # hard-coded for simplicity, CHANGEME for other datasets
            # Mid_Air : 1024 * 1024
            img_size = np.array([1024, 1024])

            if blur:
                blurer = image_distort.BlurDistort(size=int(blur))
            if occ:
                occluder = image_distort.RectangleOcculsion(dimensions=np.array([occ, occ]), centre_point=img_size / 2)
            if res:
                reres = image_distort.ReRes(scale_factor=res)

            glarer = image_distort.GlareDistort()

            for topic, msg, t in self.input_bag.read_messages():
                # if re.findall("cam0", topic):
                if re.findall("camera", topic):
                    img = self.imgmsg_to_cv2(msg)

                    if disp:
                        cv2.imshow("input", img)
                        cv2.waitKey(1)

                    if blur:
                        img = blurer.draw(img)
                    if occ:
                        img = occluder.draw(img)
                    if res:
                        img = reres.draw(img)

                    img = glarer.draw(img)

                    img_out = img

                    if disp:
                        cv2.imshow("output", img_out)
                        cv2.waitKey(1)

                    msg_out = msg
                    msg_out.data = self.cv2_to_imgmsg(img_out, encoding="mono8").data
                    (msg_out.height, msg_out.width,channal) = img_out.shape
                    msg_out.step = msg_out.width * 3  # only for mono8 images, *3 for rgb

                    self.output_bag.write(topic, msg_out, t)
                #
                else:  # if re.findall("imu0", topic):
                    self.output_bag.write(topic, msg, t)

    def view_output(self, frame_delay=1):
        # scan through output_bag to verify correct storage
        with rosbag.Bag(self.output_bagname, 'r') as self.output_bag:
            for topic, msg, t in self.output_bag.read_messages():
                # if re.findall("cam0", topic):
                if re.findall("camera", topic):
                    img = self.imgmsg_to_cv2(msg)
                    cv2.imshow("verify", img)
                    cv2.waitKey(frame_delay)

    def save_frame(self, frame_time=25):
        # scan through output_bag to verify correct storage
        t0 = None
        with rosbag.Bag(self.output_bagname, 'r') as self.output_bag:
            for topic, msg, t in self.output_bag.read_messages():
                if not t0:
                    t0 = t

                if t.secs - t0.secs >= frame_time:
                    # if re.findall("cam0", topic):
                    if re.findall("camera", topic):
                        img = self.imgmsg_to_cv2(msg)
                        cv2.imwrite("capture.png", img)
                        cv2.imshow("verify", img)
                        return


def multi_bag(src_file, out_file, blur=None, occ=None, res=None):
    with ReBagger(src_file, out_file) as rb:
        rb.generate_bag(blur=blur, occ=occ, res=res)
    print(f"Completed {out_file}")


if __name__ == "__main__":
    source_file = "/home/cor21cz/dataset/MidAir/Kite_training/sunny/trajectory_0001_left_and_right.bag"
    out_file = "/home/cor21cz/dataset/MidAir/Kite_training/sunny/trajectory_0001_left_and_right_m.bag"
    with ReBagger(source_file, out_file) as rb:
        # rb.view_output(10)
        rb.save_frame()
    src_file = source_file
    
    jobs = []
    
    blurs = np.arange(5, 101, 5)
    for blur in blurs:
        out_file = f"dataset/MidAir/test_blur{blur:02d}.bag"
        keywords = {"blur": blur}
        j = multiprocessing.Process(target=multi_bag, args=(src_file, out_file), kwargs=keywords)
        jobs.append(j)
    
    lengths = np.arange(50, 451, 50)
    for length in lengths:
        out_file = f"dataset/MidAir/test_occ{length:03d}^2.bag"
        keywords = {"occ": length}
        j = multiprocessing.Process(target=multi_bag, args=(src_file, out_file), kwargs=keywords)
        jobs.append(j)
    
    res_factors = np.arange(0.1, 0.91, 0.1)
    for res_factor in res_factors:
        out_file = f"dataset/MidAir/test_res{res_factor:0.2f}.bag"
        keywords = {"res": res_factor}
        j = multiprocessing.Process(target=multi_bag, args=(src_file, out_file), kwargs=keywords)
        jobs.append(j)
    
    # # out_file = f"test_bags/plain/euroc_MH_04_plain.bag"
    # keywords = {}
    # j = multiprocessing.Process(target=multi_bag, args=(src_file, out_file), kwargs=keywords)
    # jobs.append(j)

    # Uncomment if you're sure you want to run this! It will overwrite existing files, and will take a while!
    print(f"Starting jobs...")
    for job in jobs:
        job.start()
    
    print(f"Jobs started")
    for job in jobs:
        job.join()

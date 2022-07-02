"""
    @author: rory_haggart
    
    BRIEF:  This script allows you to select one of the MidAir dataset extracts
            (i.e. the environment, condition, camera, and trajectory) to 
            convert into a ROS .bag file. 
            It will prompt you to select the extract and then use the 
            'sensor_records.hdf5' file and the images of this extract to 
            generate the .bag file.
            If you would like to use this on a VI-SLAM algorithm, I'd 
            recommend first running 'midair_generateIMUData.py' first, in order
            to re-generate the sensor data with consistent sensor parameters.
    
    TODO:
        Work through the ground truth attitude -> local attitude process
"""

# for bagification
import rosbag
# to store the IMU/INS measurements as a ROS message
from sensor_msgs.msg import Imu
# for image interpretation and manipulation
import cv2
# to convert from a cv2 image to a ROS compatible image data
from cv_bridge import CvBridge
# to read .hdf5 sensor records files
import h5py
# for file path interpretation and representation    
import os
# for timestamp operations
import math
# for exit control
import sys
# for some file representation stuff
from pathlib import Path

def main(*args):
    
    # the input argument is the path to the bag file that is being run in this test
    if len(args) == 0:
        # enter function to ask for specific trajectory to bagify and return selection
        [environment, condition, trajectory, camera] = userInput()  
    else:
        environment = args[0]
        condition = args[1]
        trajectory = args[2]
        camera = args[3]
    
    # use the selection to find the appropriate sensor records file
    # sensorRecordsPath = os.path.abspath(os.getcwd() + "/dataset/MidAir/" + environment + '/' + condition)
    sensorRecordsPath = os.path.abspath(os.getcwd() + '/' + environment + '/' + condition)
    # print("sensorRecordsPath=",sensorRecordsPath)
    sensorRecordsFile = sensorRecordsPath + "/sensor_records.hdf5"
    
    # if the sensor records file doesn't exist, or is not yet unzipped, exit
    if not os.path.exists(sensorRecordsFile):
        if os.path.exists(sensorRecordsPath + "/sensor_records.zip"):
            print("I did not find the file: " + sensorRecordsFile + "\n\n I did find the corresponding .zip file, however. Please uncompress this file and try again.")
        else:
            print("I did not find the file: " + sensorRecordsFile)
        sys.exit(0)
        
    # open sensor_records.hdf5
    f1 = h5py.File((sensorRecordsFile),'r+')
    
    # get imu readings plus the attitude of the vehicle
    accelerometer = f1['trajectory_' + trajectory]['imu']['accelerometer']
    gyroscope = f1['trajectory_' + trajectory]['imu']['gyroscope']
    groundTruthAttitude = f1['trajectory_' + trajectory]['groundtruth']['attitude']
    
    # get the accelerometer data from the sensor records (m/s^2)
    accelerometerData = list(accelerometer) 
    # get the gyroscope data from the sensor records (rad/s)
    gyroscopeData = list(gyroscope) 
    # list the relative paths of the images for the selected camera
    imagePaths = list(f1['trajectory_' + trajectory]['camera_data'][camera]) 
    
    # exit if the selected trajectory images haven't been unzipped yet
    if (not any('.JPEG' in a for a in os.listdir(sensorRecordsPath + "/" + camera + "/trajectory_" + trajectory))) and any('.zip' in a for a in os.listdir(sensorRecordsPath + "/" + camera + "/trajectory_" + trajectory)):
        print("The images for this particular trajectory have not yet been unzipped.\nPlease unzip and try again.")
        sys.exit(0)
    
    # .bag file name is of the format trajectoryNumber_camera.bag and is located in environment/condition
    bagFilePath = sensorRecordsPath + "/trajectory_" + trajectory + "_" + camera + ".bag"
    
    # just check the user is okay if overwriting a bag that already exists
    if(os.path.isfile(bagFilePath)):
        answer = input("The .bag file {} already exists. Would you like to overwrite? (y/n)\n".format(bagFilePath))
        if(answer=='n' or answer=='N'):
            sys.exit(0)
    
    # open our .bag file to write to
    bag = rosbag.Bag(bagFilePath, "w")
    
    # initialise sequence number
    seq = 0
    # define camera frame rate
    cameraRate = 25;
    # an arbitrary starting time (seconds since epoch). mostly protecting against the invalid '0' time
    initialTime = 100000;
    
    # image dimensions
    imHeight = 1024
    imWidth = 1024
    
    print("")
    
    for line in imagePaths:
        line = str(line)
        line = line[2:-1]
        absImg = sensorRecordsPath + "/" + line # get the absolute file path
        print("absolute file path=",absImg)
        
        # ---------- IMAGE RECORD ---------- #
        # if the image exists and isn't a dud
        if (os.path.isfile(absImg)) and (os.stat(absImg).st_size != 0):
    
            print("\rAdding Image: {}/{}    ".format(seq+1, len(imagePaths)), end='')
    
            # read the image
            image = cv2.imread(absImg)
            
            # convert from cv2 to ROS by creating an Image(). This auto allocates
                # the image data to the .data field so headers/config can be
                # added below
            bridge = CvBridge()
            #img_msg.data = image
            img_msg = bridge.cv2_to_imgmsg(image, "passthrough")
            
            # img_msg format: 
            # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html
            
            # get the timestamp of this frame
            timestamp = seq * 1/cameraRate + initialTime
            # the seconds part of the timestamp is just the whole number
            timestampSec = math.floor(timestamp)
            # the nanoseconds part of the timestamp is then the decimal part
            timestampnSec = int(round(timestamp - timestampSec, 3) * pow(10, 9))
            
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
            bag.write("camera/image_raw", img_msg, img_msg.header.stamp)
            
            # increment seq num
            seq = seq + 1
        else:
            print("Could not find %s" % absImg[absImg.rindex('/') + 1:])
            
    # reset sequence ID for IMU recording
    seq = 0
    
    # the update rate of the IMU is 100Hz
    imuRate = 100
    
    print("\nAdding IMU Data")
    
    for accelLine, gyroLine, attitudeLine in zip(accelerometerData, gyroscopeData, groundTruthAttitude):
        
        # get the timestamp of this frame
        timestamp = seq * 1/imuRate + initialTime
        # the seconds part of the timestamp is just the whole number
        timestampSec = math.floor(timestamp)
        # the nanoseconds part of the timestamp is then the decimal part
        timestampnSec = int(round(timestamp - timestampSec, 3) * pow(10, 9))
        
        # create an imu_msg for our inertial data
        imu_msg = Imu()
        
        # imu_msg format: 
        # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html
        
        # ---------- IMU RECORD ---------- #
        # header info
        imu_msg.header.frame_id = "imu0"
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
        
        # attitude (quaternion)
        imu_msg.orientation.w = attitudeLine[0]
        imu_msg.orientation.x = attitudeLine[1]
        imu_msg.orientation.y = attitudeLine[2]
        imu_msg.orientation.z = attitudeLine[3]
        
        # write the imu_msg to the bag file
        bag.write("imu", imu_msg, imu_msg.header.stamp) 
    
        # increment the sequence counter
        seq = seq + 1
    
    # close the .bag file
    bag.close()
    # close the .hdf5 file
    f1.close()


# global flags for selection prompts
installFlag = ""
notInstallFlag = "(NOT INSTALLED)"

# prompt the user through the process of selecting the trajectory to bagify
def userInput():
    # get the path to the dataset folder
    # dataPath = os.getcwd() + "/dataset/MidAir"
    dataPath = os.getcwd()
    
    # the user will be told if particular options aren't available on their machine
    kiteTestFlag = installFlag if os.path.isdir(dataPath + "/Kite_test") else notInstallFlag
    kiteTrainFlag = installFlag if os.path.isdir(dataPath + "/Kite_training") else notInstallFlag
    pleTestFlag = installFlag if os.path.isdir(dataPath + "/PLE_test") else notInstallFlag
    pleTrainFlag = installFlag if os.path.isdir(dataPath + "/PLE_training") else notInstallFlag
    voTestFlag = installFlag if os.path.isdir(dataPath + "/VO_test") else notInstallFlag
    
    # ask for the environment to test in, noting which are not available
    answer = int(input("""Please enter the environment you are testing in:\n
    1. Kite_test {}
    2. Kite_training {}               
    3. PLE_test {}
    4. PLE_training {}
    5. VO_test {}\n\n""".format(kiteTestFlag, kiteTrainFlag,pleTestFlag,pleTrainFlag,voTestFlag)))
    
    # apply selection
    if (answer==1):
        if kiteTestFlag == notInstallFlag:
            print("Environment not installed")
            sys.exit(0)
        else:
            environment="Kite_test"
    elif(answer==2):
        if kiteTrainFlag == notInstallFlag:
            print("Environment not installed")
            sys.exit(0)
        else:
            environment="Kite_training"
    elif(answer==3):
        if pleTestFlag == notInstallFlag:
            print("Environment not installed")
            sys.exit(0)
        else:
            environment="PLE_test"
    elif(answer==4):
        if pleTrainFlag == notInstallFlag:
            print("Environment not installed")
            sys.exit(0)
        else:
            environment="PLE_training"
    elif(answer==5):
        if voTestFlag == notInstallFlag:
            print("Environment not installed")
            sys.exit(0)
        else:
            environment="VO_test"
    else:
        sys.exit("You entered an out-of-range value")
    
    # each environment is numbered and ordered slightly differently, so account for this
    if "Kite" in environment:
        # the test environment has less trajectories than the training one
        trajRange = 4 if("test" in environment) else 29
        
        # again, notify user if particular conditions aren't installed
        cloudyFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "cloudy") else notInstallFlag
        foggyFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "foggy") else notInstallFlag
        sunnyFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "sunny") else notInstallFlag
        sunsetFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "sunset") else notInstallFlag
        
        # ask the user which condition they'd like to test under
        answer = int(input("""Please enter the condition you are testing in:\n
    1. cloudy {}
    2. foggy {}           
    3. sunny {}
    4. sunset {}\n\n""".format(cloudyFlag,foggyFlag,sunnyFlag,sunsetFlag)))
        if (answer==1):
            if cloudyFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="cloudy"
                trajectoryLead = "3" 
        elif(answer==2):
            if foggyFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="foggy"
                trajectoryLead = "2"
        elif(answer==3):
            if sunnyFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="sunny"
                trajectoryLead = "0"
        elif(answer==4):
            if sunsetFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="sunset"
                trajectoryLead = "1"
        else:
            sys.exit(0)
         
        # look for available trajectories at this path
        trajSearchPath = dataPath + "/" + environment + "/" + condition
        # get the camera and trajectory number from the user
        trajNo, camera = trajPrinter(trajSearchPath, trajRange)
        # exit if not an existing trajectory
        if(trajNo > trajRange or trajNo < 0):
            sys.exit("You entered an out-of-range value")   
        
        # different conditions append a leading digit to the number - add this
        trajectory = trajectoryLead + str(trajNo).zfill(3)
            
    elif "PLE" in environment:
        # number of trajectories for the test and train sets
        trajRange = 5 if("test" in environment) else 23
        
        # notify of unavailable conditions
        fallFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "fall") else notInstallFlag
        springFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "spring") else notInstallFlag
        winterFlag = installFlag if os.path.isdir(dataPath + "/" + environment + "/" + "winter") else notInstallFlag
        
        # ask for the condition to test under
        answer = int(input("""Please enter the condition you are testing in:\n
    1. fall {}
    2. spring {}              
    3. winter {}\n\n""".format(fallFlag,springFlag,winterFlag)))
        if (answer==1):
            if fallFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="fall"
        elif(answer==2):
            if springFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="spring"
        elif(answer==3):
            if winterFlag == notInstallFlag:
                print("Condition not installed")
                sys.exit(0)
            else:
                condition="winter"
        else:
            sys.exit(0)    
        
        # get the camera to use and the trajectory to test
        trajSearchPath = dataPath + "/" + environment + "/" + condition
        trajNo, camera = trajPrinter(trajSearchPath, trajRange)
        if(trajNo > trajRange or trajNo < 0):
            sys.exit("You entered an out-of-range value")
        trajectory = "4" + str(trajNo).zfill(3)
                
    elif(environment=="VO_test"):
        trajRange = 2
        
        answer = int(input("""Please enter the condition you are testing in:\n
    1. foggy               
    2. sunny
    3. sunset\n\n"""))
        if(answer==1):
            condition="foggy"
            trajectoryLead = "1"
        elif(answer==2):
            condition="sunny"
            trajectoryLead = "0"
        elif(answer==3):
            condition="sunset"
            trajectoryLead = "2"
        else:
            sys.exit("You entered an invalid value")
            
        trajSearchPath = dataPath + "/" + environment + "/" + condition
        trajNo, camera = trajPrinter(trajSearchPath, trajRange)
        if(trajNo > trajRange or trajNo < 0):
            sys.exit("You entered an out-of-range value")
        trajectory = trajectoryLead + str(trajNo).zfill(3)
        
    return [environment, condition, trajectory, camera]
    
# print trajectory numbers with notice of whether or not they are installed
def trajPrinter(trajSearchPath, trajRange):
    
    colorLeftFlag =  installFlag if os.path.isdir(trajSearchPath + "/color_left") else notInstallFlag   
    colorRightFlag =  installFlag if os.path.isdir(trajSearchPath + "/color_right") else notInstallFlag 
    colorDownFlag =  installFlag if os.path.isdir(trajSearchPath + "/color_down") else notInstallFlag 
    
    answer = int(input("""Please enter the camera you are testing with:\n
    1. color_left {}
    2. color_right {}             
    3. color_down {} \n\n""".format(colorLeftFlag,colorRightFlag,colorDownFlag)))
    
    if (answer==1):
        if colorLeftFlag == notInstallFlag:
            print("Camera not installed")
            sys.exit(0)
        else:
            camera="color_left"
    elif(answer==2):
        if colorRightFlag == notInstallFlag:
            print("Camera not installed")
            sys.exit(0)
        else:
            camera="color_right"
    elif(answer==3):
        if colorDownFlag == notInstallFlag:
            print("Camera not installed")
            sys.exit(0)
        else:
            camera="color_down"
    else:
        sys.exit("You entered an invalid value") 
    
    trajSearchPath = trajSearchPath + "/" + camera
    
    trajFileList = list(Path(trajSearchPath).rglob("[trajectory]*"))
    trajFileList = [str(a) for a in trajFileList if ("trajectory" in str(a) and ".bag" not in str(a))]
    trajList = [int(a[-2:]) for a in trajFileList]
    
    zippedFlag = "(NOT UNZIPPED)"
    
    print("Please select the trajectory to test:\n")
    for i in range(trajRange + 1):
        trajFlag = installFlag if i in trajList else notInstallFlag
        trajFolder = [s for s in trajFileList if s[-3:] == ("0" + str(i).zfill(2))]
        if len(trajFolder) != 0:
            if (not any('.JPEG' in a for a in os.listdir(trajFolder[0]))) and any('.zip' in a for a in os.listdir(trajFolder[0])):
                trajFlag = zippedFlag
        print("    {}. {}".format(i, trajFlag))
        
    trajNo = int(input(""))
    
    if trajNo not in trajList and trajNo <= trajRange and trajNo >= 0:
        print("Trajectory not installed")
        sys.exit(0)
    
    print("trajNo=",trajNo)
    print("camera=",camera)
    while 1:
        a=1
    return(trajNo, camera)

if __name__ == "__main__":
    main()
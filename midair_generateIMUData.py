"""
Adaption of work by:
Michael Fonder and Marc Van Droogenbroeck
 Mid-Air: A multi-modal dataset for extremely low altitude drone flights
 Conference on Computer Vision and Pattern Recognition Workshop (CVPRW)

BRIEF:  Since the IMU data in the midair dataset is simulated, the data
        must be synthesised using the 'ground truth' data. This file
        does this and allows for easy customisability fo the sensor noise
        parameters, which allows for accurate representation in VI-SLAM 
        configuration
        
TODO:
    ensure portability and readability 
    assess random walk - is this currently just like more gaussian noise
        and not a cumulative thing? needs more research
"""
import h5py
import numpy as np
from pyquaternion import Quaternion
import os
import sys

if __name__ == '__main__':
    answer = int(input("""Please enter the environment you are testing in:\n
    1. Kite_test
    2. Kite_training               
    3. PLE_test
    4. PLE_training
    5. VO_test\n\n"""))
    
    if (answer==1):
        environment="Kite_test"
    elif(answer==2):
        environment="Kite_training"
    elif(answer==3):
        environment="PLE_test"
    elif(answer==4):
        environment="PLE_training"
    elif(answer==5):
        environment="VO_test"
    else:
        sys.exit("You entered an invalid value")
        
    if "Kite" in environment:
        answer = int(input("""Please enter the condition you are testing in:\n
    1. cloudy
    2. foggy               
    3. sunny
    4. sunset\n\n"""))
        if (answer==1):
            condition="cloudy"
        elif(answer==2):
            condition="foggy"
        elif(answer==3):
            condition="sunny"
        elif(answer==4):
            condition="sunset"
        else:
            sys.exit("You entered an invalid value")
    elif "PLE" in environment:
        answer = int(input("""Please enter the condition you are testing in:\n
    1. fall
    2. spring               
    3. winter\n\n"""))
        if (answer==1):
            condition="fall"
        elif(answer==2):
            condition="spring"
        elif(answer==3):
            condition="winter"
        else:
            sys.exit("You entered an invalid value")    
    elif(environment=="VO_test"):        
        answer = int(input("""Please enter the condition you are testing in:\n
    1. foggy               
    2. sunny
    3. sunset\n\n"""))
        if(answer==1):
            condition="foggy"
        elif(answer==2):
            condition="sunny"
        elif(answer==3):
            condition="sunset"
        else:
            sys.exit("You entered an invalid value")

    accStdDev = 0.08        # [m/s^2] standard deviation of accelerometer noise (gaussian)
    gyrStdDev = 0.004       # [rad/s] standard deviation of gyroscope noise (gaussian)
    accRWStdDev = 0.00004   # [m/s^2] accelerometer bias random walk noise standard deviation 
    gyrRWStdDev = 2.0e-6    # [rad/s] gyroscope bias random walk noise standard deviation 
    
    print("""Default Sensor Parameters:\n
    Accelerometer Noise Standard Deviation: {}
    Gyroscope Noise Standard Deviation: {}
    Accelerometer Bias Random Walk Noise Standard Deviation: {}
    Gyroscope Bias Random Walk Noise Standard Deviation: {}\n""".format(accStdDev,gyrStdDev,accRWStdDev,gyrRWStdDev))
    
    answer = str(input("""Would you like to use the default sensor parameters? (y/n):\n\n"""))
    
    if(answer=="n" or answer=="N"):
        print("Using custom sensor parameters.\n")
        print("WARN: before you process this data in a VI-SLAM algorithm, make sure you update the appropriate .yaml file with the new IMU values.\n")
        accStdDev = input("Accelerometer Noise Standard Deviation: ")
        gyrStdDev = input("Gyroscope Noise Standard Deviation: ")
        accRWStdDev = input("Accelerometer Bias Random Walk Noise Standard Deviation: ")
        gyrRWStdDev = input("Gyroscope Bias Random Walk Noise Standard Deviation: ")
        
        try:
            accStdDev = float(accStdDev)
            gyrStdDev = float(gyrStdDev)
            accRWStdDev = float(accRWStdDev)
            gyrRWStdDev = float(gyrRWStdDev)
        except:
            sys.exit("You entered a non-numeric sensor parameter")
        
    elif(answer=="y" or answer=="Y"):
        print("Using default sensor parameters.\n")
    else:
        sys.exit("You entered an invalid value")
    
    # define the path to the folder containing our sensor records
    sensorRecords = os.getcwd() + '/' + environment + '/' + condition + '/sensor_records.hdf5'
    
    assert os.path.exists(sensorRecords), "\nI did not find the file: " + sensorRecords + """\n\nThis could mean that you haven't downloaded this segment, or that the sensor_records file hasn't been unzipped yet"""
    
    answer = str(input("Warning: this script will overwrite IMU measurements stored in the given hdf5 dataset. \n"+ \
                       "Do you want to proceed? (y/n): "))
    if not(answer=="y" or answer=="Y"):
        sys.exit(0)

    database = h5py.File(sensorRecords, "a")
    db_path = os.path.dirname(sensorRecords)

    # IMU noise parameters chosen randomly in a range of values encountered in real devices
    #noise_acc = 2 * np.power(10., -np.random.uniform(low=1., high=3., size=(1, 3))) 
    noise_acc = np.array([[accStdDev]*3])#2 * np.power(10., [[-1*np.random.randint(10000, 30000) / 10000] * 3])
    
    #noise_gyr = np.power(10., -np.random.uniform(low=1., high=3., size=(1, 3)))
    noise_gyr = np.array([[gyrStdDev]*3])#np.power(10., [[-1*np.random.randint(10000, 30000) / 10000] * 3])
    
    #imu_bias_acc_rw = 2 * np.power(10., -np.random.uniform(low=3., high=6., size=(1, 3)))
    imu_bias_acc_rw = np.array([[accRWStdDev]*3])#2 * np.power(10., [[-1*np.random.randint(30000, 60000) / 10000] * 3])
    
    #imu_bias_gyr_rw = np.power(10., -np.random.uniform(low=4., high=6., size=(1, 3)))
    imu_bias_gyr_rw = np.array([[gyrRWStdDev]*3]) #np.power(10., [[-1*np.random.randint(40000, 60000) / 10000] * 3])
    
    for dataset in database:
        print("Currently processing : %s" % dataset)
        gt_group = database[dataset]["groundtruth"]
        gt_attitude = gt_group["attitude"]
        gt_angular_vel = gt_group["angular_velocity"]
        gt_accelerations = gt_group["acceleration"]

        imu_group = database[dataset]["imu"]

        # Set init parameters
        imu_accelerometer = np.zeros(gt_attitude.shape, dtype=float)
        imu_gyroscope = np.zeros(gt_attitude.shape, dtype=float)

        imu_bias_acc = np.array([[0.,0.,0.]])#np.random.normal([0., 0., 0.], imu_bias_acc_rw)
        imu_bias_gyr = np.array([[0.,0.,0.]])#np.random.normal([0., 0., 0.], imu_bias_gyr_rw)

        init_bias_est_acc = imu_bias_acc + np.random.normal([0., 0., 0.], noise_acc)
        init_bias_est_gyr = imu_bias_gyr + np.random.normal([0., 0., 0.], noise_gyr)
        
        imu_group["accelerometer"].attrs["init_bias_est"] = init_bias_est_acc
        imu_group["gyroscope"].attrs["init_bias_est"] = init_bias_est_gyr
        
        
        # Pass over trajectory to generate simulated sensor measurements
        for i in range(gt_attitude.shape[0]):
            attitude = Quaternion(gt_attitude[i, :])
            
            accNoiseInstant = np.random.normal([0., 0., 0.], noise_acc)
            gyrNoiseInstant = np.random.normal([0., 0., 0.], noise_gyr)
            
            imu_accelerometer = attitude.conjugate.rotate(gt_accelerations[i, :] + np.array([0., 0., -9.81])) \
                                                            + imu_bias_acc + accNoiseInstant
            imu_gyroscope = gt_angular_vel[i, :] + imu_bias_gyr + gyrNoiseInstant
            
            accRWComponent = np.random.normal([0., 0., 0.], imu_bias_acc_rw)
            gyrRWComponent = np.random.normal([0., 0., 0.], imu_bias_gyr_rw)
            
            imu_bias_acc += accRWComponent
            imu_bias_gyr += gyrRWComponent
            
            imu_group["accelerometer"][i] = imu_accelerometer
            imu_group["gyroscope"][i] = imu_gyroscope
            
            
database.close()

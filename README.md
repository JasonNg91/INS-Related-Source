# INS-Related-Source
Inertial navigation related source

# AI-based inertial navigation source
## 1. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods
- paper:https://paperswithcode.com/paper/ronin-robust-neural-inertial-navigation-in
- github: (1) https://github.com/Sachini/ronin (2) https://github.com/BehnamZeinali/IMUNet

This paper sets a new foundation for data-driven inertial navigation research, where the task is the estimation of positions and orientations of a moving subject from a sequence of IMU sensor measurements. More concretely, the paper presents 1) a new benchmark containing more than 40 hours of IMU sensor data from 100 human subjects with ground-truth 3D trajectories under natural human motions; 2) novel neural inertial navigation architectures, making significant improvements for challenging motion cases; and 3) qualitative and quantitative evaluations of the competing methods over three inertial navigation benchmarks. We will share the code and data to promote further research.

## 2. NILoc:Neural Inertial Localization
- paper: https://openaccess.thecvf.com/content/CVPR2022/html/Herath_Neural_Inertial_Localization_CVPR_2022_paper.html
- github: https://github.com/Sachini/niloc

This paper proposes the inertial localization problem, the task of estimating the absolute location from a sequence of inertial sensor measurements. This is an exciting and unexplored area of indoor localization research, where we present a rich dataset with 53 hours of inertial sensor data and the associated ground truth locations. We developed a solution, dubbed neural inertial localization (NILoc) which 1) uses a neural inertial navigation technique to turn inertial sensor history to a sequence of velocity vectors; then 2) employs a transformer-based neural architecture to find the device location from the sequence of velocity estimates. We only use an IMU sensor, which is energy efficient and privacy-preserving compared to WiFi, cameras, and other data sources. Our approach is significantly faster and achieves competitive results even compared with state-of-the-art methods that require a floorplan and run 20 to 30 times slower. We share our code, model and data at https://sachini.github.io/niloc.

## 3. Deep Learning Based Speed Estimation for Constraining Strapdown Inertial Navigation on Smartphones
- paper: https://paperswithcode.com/paper/deep-learning-based-speed-estimation-for
- github: https://github.com/AaltoVision/deep-speed-constrained-ins

Strapdown inertial navigation systems are sensitive to the quality of the data provided by the accelerometer and gyroscope. Low-grade IMUs in handheld smart-devices pose a problem for inertial odometry on these devices. We propose a scheme for constraining the inertial odometry problem by complementing non-linear state estimation by a CNN-based deep-learning model for inferring the momentary speed based on a window of IMU samples. We show the feasibility of the model using a wide range of data from an iPhone, and present proof-of-concept results for how the model can be combined with an inertial navigation system for three-dimensional inertial navigation.

## 4. RINS-W: Robust Inertial Navigation System on Wheels
- paper: https://cs.paperswithcode.com/paper/rins-w-robust-inertial-navigation-system-on
- github: https://github.com/mbrossar/RINS-W
  
This paper proposes a real-time approach for long-term inertial navigation based only on an Inertial Measurement Unit (IMU) for self-localizing wheeled robots. The approach builds upon two components: 1) a robust detector that uses recurrent deep neural networks to dynamically detect a variety of situations of interest, such as zero velocity or no lateral slip; and 2) a state-of-the-art Kalman filter which incorporates this knowledge as pseudo-measurements for localization. Evaluations on a publicly available car dataset demonstrates that the proposed scheme may achieve a final precision of 20 m for a 21 km long trajectory of a vehicle driving for over an hour, equipped with an IMU of moderate precision (the gyro drift rate is 10 deg/h). To our knowledge, this is the first paper which combines sophisticated deep learning techniques with state-of-the-art filtering methods for pure inertial navigation on wheeled vehicles and as such opens up for novel data-driven inertial navigation techniques. Moreover, albeit taylored for IMU-only based localization, our method may be used as a component for self-localization of wheeled robots equipped with a more complete sensor suite.

## 5-1. Improving Foot-Mounted Inertial Navigation Through Real-Time Motion Classification  
- paper: https://cs.paperswithcode.com/paper/improving-foot-mounted-inertial-navigation
- github: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a foot-mounted, zero-velocity-aided inertial navigation system (INS) by varying estimator parameters based on a real-time classification of motion type. We train a support vector machine (SVM) classifier using inertial data recorded by a single foot-mounted sensor to differentiate between six motion types (walking, jogging, running, sprinting, crouch-walking, and ladder-climbing) and report mean test classification accuracy of over 90% on a dataset with five different subjects. From these motion types, we select two of the most common (walking and running), and describe a method to compute optimal zero-velocity detection parameters tailored to both a specific user and motion type by maximizing the detector F-score. By combining the motion classifier with a set of optimal detection parameters, we show how we can reduce INS position error during mixed walking and running motion. We evaluate our adaptive system on a total of 5.9 km of indoor pedestrian navigation performed by five different subjects moving along a 130 m path with surveyed ground truth markers.

## 5-2. LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation
- paper: https://cs.paperswithcode.com/paper/lstm-based-zero-velocity-detection-for-robust
- github: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a zero-velocity-aided inertial navigation system (INS) by replacing the standard zero-velocity detector with a long short-term memory (LSTM) neural network. While existing threshold-based zero-velocity detectors are not robust to varying motion types, our learned model accurately detects stationary periods of the inertial measurement unit (IMU) despite changes in the motion of the user. Upon detection, zero-velocity pseudo-measurements are fused with a dead reckoning motion model in an extended Kalman filter (EKF). We demonstrate that our LSTM-based zero-velocity detector, used within a zero-velocity-aided INS, improves zero-velocity detection during human localization tasks. Consequently, localization accuracy is also improved. Our system is evaluated on more than 7.5 km of indoor pedestrian locomotion data, acquired from five different subjects. We show that 3D positioning error is reduced by over 34% compared to existing fixed-threshold zero-velocity detectors for walking, running, and stair climbing motions. Additionally, we demonstrate how our learned zero-velocity detector operates effectively during crawling and ladder climbing. Our system is calibration-free (no careful threshold-tuning is required) and operates consistently with differing users, IMU placements, and shoe types, while being compatible with any generic zero-velocity-aided INS.

## 5-3. Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation  
- paper: https://cs.paperswithcode.com/paper/robust-data-driven-zero-velocity-detection 
- github: https://github.com/utiasSTARS/pyshoe

We present two novel techniques for detecting zero-velocity events to improve foot-mounted inertial navigation. Our first technique augments a classical zero-velocity detector by incorporating a motion classifier that adaptively updates the detector's threshold parameter. Our second technique uses a long short-term memory (LSTM) recurrent neural network to classify zero-velocity events from raw inertial data, in contrast to the majority of zero-velocity detection methods that rely on basic statistical hypothesis testing. We demonstrate that both of our proposed detectors achieve higher accuracies than existing detectors for trajectories including walking, running, and stair-climbing motions. Additionally, we present a straightforward data augmentation method that is able to extend the LSTM-based model to different inertial sensors without the need to collect new training data.


## 6. GPS-Denied Navigation Using Low-Cost Inertial Sensors and Recurrent Neural Networks
- paper: https://paperswithcode.com/paper/gps-denied-navigation-using-low-cost-inertial
- github: https://github.com/majuid/DeepNav

Autonomous missions of drones require continuous and reliable estimates for the drone's attitude, velocity, and position. Traditionally, these states are estimated by applying Extended Kalman Filter (EKF) to Accelerometer, Gyroscope, Barometer, Magnetometer, and GPS measurements. When the GPS signal is lost, position and velocity estimates deteriorate quickly, especially when using low-cost inertial sensors. This paper proposes an estimation method that uses a Recurrent Neural Network (RNN) to allow reliable estimation of a drone's position and velocity in the absence of GPS signal. The RNN is trained on a public dataset collected using Pixhawk. This low-cost commercial autopilot logs the raw sensor measurements (network inputs) and corresponding EKF estimates (ground truth outputs). The dataset is comprised of 548 different flight logs with flight durations ranging from 4 to 32 minutes. For training, 465 flights are used, totaling 45 hours. The remaining 83 flights totaling 8 hours are held out for validation. Error in a single flight is taken to be the maximum absolute difference in 3D position (MPE) between the RNN predictions (without GPS) and the ground truth (EKF with GPS). On the validation set, the median MPE is 35 meters. MPE values as low as 2.7 meters in a 5-minutes flight could be achieved using the proposed method. The MPE in 90% of the validation flights is bounded below 166 meters. The network was experimentally tested and worked in real-time.


## 7. CTIN: Robust Contextual Transformer Network for Inertial Navigation
- paper: https://paperswithcode.com/paper/ctin-robust-contextual-transformer-network
- github: https://github.com/bingrao/ctin

Recently, data-driven inertial navigation approaches have demonstrated their capability of using well-trained neural networks to obtain accurate position estimates from inertial measurement units (IMU) measurements. In this paper, we propose a novel robust Contextual Transformer-based network for Inertial Navigation~(CTIN) to accurately predict velocity and trajectory. To this end, we first design a ResNet-based encoder enhanced by local and global multi-head self-attention to capture spatial contextual information from IMU measurements. Then we fuse these spatial representations with temporal knowledge by leveraging multi-head attention in the Transformer decoder. Finally, multi-task learning with uncertainty reduction is leveraged to improve learning efficiency and prediction accuracy of velocity and trajectory. Through extensive experiments over a wide range of inertial datasets~(e.g. RIDI, OxIOD, RoNIN, IDOL, and our own), CTIN is very robust and outperforms state-of-the-art models.

## 8. AI-IMU Dead-Reckoning
- paper: https://ieeexplore.ieee.org/document/9035481
- github: https://github.com/mbrossar/ai-imu-dr

In this paper, we propose a novel accurate method for dead-reckoning of wheeled vehicles based only on an Inertial Measurement Unit (IMU). In the context of intelligent vehicles, robust and accurate dead-reckoning based on the IMU may prove useful to correlate feeds from imaging sensors, to safely navigate through obstructions, or for safe emergency stops in the extreme case of exteroceptive sensors failure. The key components of the method are the Kalman filter and the use of deep neural networks to dynamically adapt the noise parameters of the filter. The method is tested on the KITTI odometry dataset, and our dead-reckoning inertial method based only on the IMU accurately estimates 3D position, velocity, orientation of the vehicle and self-calibrates the IMU biases. We achieve on average a 1.10% translational error and the algorithm competes with top-ranked methods which, by contrast, use LiDAR or stereo vision.

## 9. End-to-End Learning Framework for IMU-Based 6-DOF Odometry
- paper: https://www.mdpi.com/1424-8220/19/17/3777
- github: https://github.com/jpsml/6-DOF-Inertial-Odometry

This paper presents an end-to-end learning framework for performing 6-DOF odometry by using only inertial data obtained from a low-cost IMU. The proposed inertial odometry method allows leveraging inertial sensors that are widely available on mobile platforms for estimating their 3D trajectories. For this purpose, neural networks based on convolutional layers combined with a two-layer stacked bidirectional LSTM are explored from the following three aspects. First, two 6-DOF relative pose representations are investigated: one based on a vector in the spherical coordinate system, and the other based on both a translation vector and an unit quaternion. Second, the loss function in the network is designed with the combination of several 6-DOF pose distance metrics: mean squared error, translation mean absolute error, quaternion multiplicative error and quaternion inner product. Third, a multi-task learning framework is integrated to automatically balance the weights of multiple metrics. In the evaluation, qualitative and quantitative analyses were conducted with publicly-available inertial odometry datasets. The best combination of the relative pose representation and the loss function was the translation and quaternion together with the translation mean absolute error and quaternion multiplicative error, which obtained more accurate results with respect to state-of-the-art inertial odometry techniques.


# New filter methods for inertial navigation
## 1. Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation
- papaer: https://cs.paperswithcode.com/paper/190409251
- github: https://github.com/RossHartley/invariant-ekf

Legged robots require knowledge of pose and velocity in order to maintain stability and execute walking paths. Current solutions either rely on vision data, which is susceptible to environmental and lighting conditions, or fusion of kinematic and contact data with measurements from an inertial measurement unit (IMU). In this work, we develop a contact-aided invariant extended Kalman filter (InEKF) using the theory of Lie groups and invariant observer design. This filter combines contact-inertial dynamics with forward kinematic corrections to estimate pose and velocity along with all current contact points. We show that the error dynamics follows a log-linear autonomous differential equation with several important consequences: (a) the observable state variables can be rendered convergent with a domain of attraction that is independent of the system's trajectory; (b) unlike the standard EKF, neither the linearized error dynamics nor the linearized observation model depend on the current state estimate, which (c) leads to improved convergence properties and (d) a local observability matrix that is consistent with the underlying nonlinear system. Furthermore, we demonstrate how to include IMU biases, add/remove contacts, and formulate both world-centric and robo-centric versions. We compare the convergence of the proposed InEKF with the commonly used quaternion-based EKF though both simulations and experiments on a Cassie-series bipedal robot. Filter accuracy is analyzed using motion capture, while a LiDAR mapping experiment provides a practical use case. Overall, the developed contact-aided InEKF provides better performance in comparison with the quaternion-based EKF as a result of exploiting symmetries present in system.


## 2. The invariant extended Kalman filter as a stable observer
- paper: The invariant extended Kalman filter as a stable observer
- github:  

    (1) manif: A small header-only library for Lie theory: https://github.com/artivis/manif \
    (2) inekf: a C++ library that implements an invariant extended Kalman filter (InEKF) for 3D aided inertial navigation: https://github.com/RossHartley/invariant-ekf \
    (3) kalmanif: A small collection of Kalman Filters on Lie groups: https://github.com/artivis/kalmanif

We analyze the convergence aspects of the invariant extended Kalman filter (IEKF), when the latter is used as a deterministic non-linear observer on Lie groups, for continuous-time systems with discrete observations. One of the main features of invariant observers for left-invariant systems on Lie groups is that the estimation error is autonomous. In this paper we first generalize this result by characterizing the (much broader) class of systems for which this property holds. Then, we leverage the result to prove for those systems the local stability of the IEKF around any trajectory, under the standard conditions of the linear case. One mobile robotics example and one inertial navigation example illustrate the interest of the approach. Simulations evidence the fact that the EKF is capable of diverging in some challenging situations, where the IEKF with identical tuning keeps converging.

# AHRS related source
## 1. Fusion: opensource AHRS code from xioTechnologies
- github: https://github.com/JasonNg91/Fusion-from-xioTechnologies-AHRS

Fusion is a sensor fusion library for Inertial Measurement Units (IMUs), optimised for embedded systems. Fusion is a C library but is also available as the Python package, imufusion. Two example Python scripts, simple_example.py and advanced_example.py are provided with example sensor data to demonstrate use of the package.

## 2. AHRS-from-Mayitzin
- github: https://github.com/JasonNg91/AHRS-from-Mayitzin

AHRS is a collection of functions and algorithms in pure Python used to estimate the orientation of mobile systems. Orginally, an AHRS is a set of orthogonal sensors providing attitude information about an aircraft. This field has now expanded to smaller devices, like wearables, automated transportation and all kinds of systems in motion. This package's focus is fast prototyping, education, testing and modularity. Performance is NOT the main goal. For optimized implementations there are endless resources in C/C++ or Fortran. AHRS is compatible with Python 3.6 and newer.

## 3. FastAHRS
- github: https://github.com/JasonNg91/FastAhrs

Robust 400Hz Atitude and Heading (AHRS) estimation for the AdaFruit "Precision NXP 9-DOF breakout board - FXOS8700 + FXAS21002". This library solves a bunch of issues that come with the default AdaFruit libraries, and uses the latest updates of the MadgWick filter that few people seem to be aware of.

# Classic inertial navigation source
## 1. grizz_ins
 - github: https://github.com/thetengda/grizz_ins

Pure Ins integrator, tested with high-precision dataset

# Magnetic field navigation related source
## 1. ros2_magslam
 - github: https://github.com/JasonNg91/ros2_magslam

These packages can be used to perform mapping of the magnetic field using Gaussian Processes in ROS2 in psuedo real-time. 


## 2. geomag
- github: https://github.com/cmweiss/geomag/tree/master/geomag

Magnetic variation/declination: Calculates magnetic variation/declination for any latitude/longitude/altitude, for any date. Uses the NOAA National Geophysical Data Center, epoch 2015 data.

# IMU simulator related source
## 1. gnss-ins-simulator-from-Aceinna
- github: https://github.com/JasonNg91/gnss-ins-simulator-from-Aceinna

GNSS-INS-SIM is an GNSS/INS simulation project, which generates reference trajectories, IMU sensor output, GPS output, odometer output and magnetometer output. Users choose/set up the sensor model, define the waypoints and provide algorithms, and gnss-ins-sim can generate required data for the algorithms, run the algorithms, plot simulation results, save simulations results, and generate a brief summary.

## 2. IMU-Simulator opensource code from xioTechnologies
- github: https://github.com/JasonNg91/IMU-Simulator-from-xioTechnologies


# Open datasets 
1- RONIN which is available at here: https://ronin.cs.sfu.ca/

2- RIDI which is available at DropBox: https://www.dropbox.com/s/9zzaj3h3u4bta23/ridi_data_publish_v2.zip?dl=0

3- OxIOD: The Dataset for Deep Inertial Odometry which is available at OxIOD: http://deepio.cs.ox.ac.uk/

4- Px4 which can be downloaded from px4 and the scripts provided here has been used to download the data and pre-process it: https://github.com/majuid/DeepNav
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
- gihub: https://github.com/mbrossar/RINS-W
  
This paper proposes a real-time approach for long-term inertial navigation based only on an Inertial Measurement Unit (IMU) for self-localizing wheeled robots. The approach builds upon two components: 1) a robust detector that uses recurrent deep neural networks to dynamically detect a variety of situations of interest, such as zero velocity or no lateral slip; and 2) a state-of-the-art Kalman filter which incorporates this knowledge as pseudo-measurements for localization. Evaluations on a publicly available car dataset demonstrates that the proposed scheme may achieve a final precision of 20 m for a 21 km long trajectory of a vehicle driving for over an hour, equipped with an IMU of moderate precision (the gyro drift rate is 10 deg/h). To our knowledge, this is the first paper which combines sophisticated deep learning techniques with state-of-the-art filtering methods for pure inertial navigation on wheeled vehicles and as such opens up for novel data-driven inertial navigation techniques. Moreover, albeit taylored for IMU-only based localization, our method may be used as a component for self-localization of wheeled robots equipped with a more complete sensor suite.

## 5-1. Improving Foot-Mounted Inertial Navigation Through Real-Time Motion Classification  
- paper: https://cs.paperswithcode.com/paper/improving-foot-mounted-inertial-navigation
- gihub: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a foot-mounted, zero-velocity-aided inertial navigation system (INS) by varying estimator parameters based on a real-time classification of motion type. We train a support vector machine (SVM) classifier using inertial data recorded by a single foot-mounted sensor to differentiate between six motion types (walking, jogging, running, sprinting, crouch-walking, and ladder-climbing) and report mean test classification accuracy of over 90% on a dataset with five different subjects. From these motion types, we select two of the most common (walking and running), and describe a method to compute optimal zero-velocity detection parameters tailored to both a specific user and motion type by maximizing the detector F-score. By combining the motion classifier with a set of optimal detection parameters, we show how we can reduce INS position error during mixed walking and running motion. We evaluate our adaptive system on a total of 5.9 km of indoor pedestrian navigation performed by five different subjects moving along a 130 m path with surveyed ground truth markers.

## 5-2. LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation
- paper: https://cs.paperswithcode.com/paper/lstm-based-zero-velocity-detection-for-robust
- gihub: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a zero-velocity-aided inertial navigation system (INS) by replacing the standard zero-velocity detector with a long short-term memory (LSTM) neural network. While existing threshold-based zero-velocity detectors are not robust to varying motion types, our learned model accurately detects stationary periods of the inertial measurement unit (IMU) despite changes in the motion of the user. Upon detection, zero-velocity pseudo-measurements are fused with a dead reckoning motion model in an extended Kalman filter (EKF). We demonstrate that our LSTM-based zero-velocity detector, used within a zero-velocity-aided INS, improves zero-velocity detection during human localization tasks. Consequently, localization accuracy is also improved. Our system is evaluated on more than 7.5 km of indoor pedestrian locomotion data, acquired from five different subjects. We show that 3D positioning error is reduced by over 34% compared to existing fixed-threshold zero-velocity detectors for walking, running, and stair climbing motions. Additionally, we demonstrate how our learned zero-velocity detector operates effectively during crawling and ladder climbing. Our system is calibration-free (no careful threshold-tuning is required) and operates consistently with differing users, IMU placements, and shoe types, while being compatible with any generic zero-velocity-aided INS.

## 5-3. Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation  
- paper: https://cs.paperswithcode.com/paper/robust-data-driven-zero-velocity-detection 
- gihub: https://github.com/utiasSTARS/pyshoe

We present two novel techniques for detecting zero-velocity events to improve foot-mounted inertial navigation. Our first technique augments a classical zero-velocity detector by incorporating a motion classifier that adaptively updates the detector's threshold parameter. Our second technique uses a long short-term memory (LSTM) recurrent neural network to classify zero-velocity events from raw inertial data, in contrast to the majority of zero-velocity detection methods that rely on basic statistical hypothesis testing. We demonstrate that both of our proposed detectors achieve higher accuracies than existing detectors for trajectories including walking, running, and stair-climbing motions. Additionally, we present a straightforward data augmentation method that is able to extend the LSTM-based model to different inertial sensors without the need to collect new training data.


## 6.  
- paper:  
- gihub: 

# Classic inertial navigation source

# IMU simulator related source

# Open datasets 

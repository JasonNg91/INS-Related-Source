# INS-Related-Source
A collection of awesome inertial navigation related source, including AI-based-INS, classic-INS, AHRS, new filter algorithm for INS, magnetic-field navigation, human activity recognization, dataset and so on.

# AI-based inertial navigation source
## 1-1. RIDI: Robust IMU Double Integration
- paper: https://paperswithcode.com/paper/ridi-robust-imu-double-integration
- github: https://github.com/higerra/ridi_imu

This paper proposes a novel data-driven approach for inertial navigation, which learns to estimate trajectories of natural human motions just from an inertial measurement unit (IMU) in every smartphone. The key observation is that human motions are repetitive and consist of a few major modes (e.g., standing, walking, or turning). Our algorithm regresses a velocity vector from the history of linear accelerations and angular velocities, then corrects low-frequency bias in the linear accelerations, which are integrated twice to estimate positions. We have acquired training data with ground-truth motions across multiple human subjects and multiple phone placements (e.g., in a bag or a hand). The qualitatively and quantitatively evaluations have demonstrated that our algorithm has surprisingly shown comparable results to full Visual Inertial navigation. To our knowledge, this paper is the first to integrate sophisticated machine learning techniques with inertial navigation, potentially opening up a new line of research in the domain of data-driven inertial navigation. We will publicly share our code and data to facilitate further research.

## 1-2. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods
- paper:https://paperswithcode.com/paper/ronin-robust-neural-inertial-navigation-in
- github: (1) https://github.com/Sachini/ronin (2) https://github.com/BehnamZeinali/IMUNet

This paper sets a new foundation for data-driven inertial navigation research, where the task is the estimation of positions and orientations of a moving subject from a sequence of IMU sensor measurements. More concretely, the paper presents 1) a new benchmark containing more than 40 hours of IMU sensor data from 100 human subjects with ground-truth 3D trajectories under natural human motions; 2) novel neural inertial navigation architectures, making significant improvements for challenging motion cases; and 3) qualitative and quantitative evaluations of the competing methods over three inertial navigation benchmarks. We will share the code and data to promote further research.

## 2. IMUNet: Efficient Regression Architecture for IMU Navigation and Positioning
- paper: https://arxiv.org/abs/2208.00068
- github: https://github.com/BehnamZeinali/IMUNet?tab=readme-ov-file

Data-driven based method for navigation and positioning has absorbed attention in recent years and it outperforms all its competitor methods in terms of accuracy and efficiency. This paper introduces a new architecture called IMUNet which is accurate and efficient for position estimation on edge device implementation receiving a sequence of raw IMU measurements. The architecture has been compared with one dimension version of the state-of-the-art CNN networks that have been introduced recently for edge device implementation in terms of accuracy and efficiency. Moreover, a new method for collecting a dataset using IMU sensors on cell phones and Google ARCore API has been proposed and a publicly available dataset has been recorded. A comprehensive evaluation using four different datasets as well as the proposed dataset and real device implementation has been done to prove the performance of the architecture. All the code in both Pytorch and Tensorflow framework as well as the Android application code have been shared to improve further research.

## 3. NILoc:Neural Inertial Localization
- paper: https://openaccess.thecvf.com/content/CVPR2022/html/Herath_Neural_Inertial_Localization_CVPR_2022_paper.html
- github: https://github.com/Sachini/niloc

This paper proposes the inertial localization problem, the task of estimating the absolute location from a sequence of inertial sensor measurements. This is an exciting and unexplored area of indoor localization research, where we present a rich dataset with 53 hours of inertial sensor data and the associated ground truth locations. We developed a solution, dubbed neural inertial localization (NILoc) which 1) uses a neural inertial navigation technique to turn inertial sensor history to a sequence of velocity vectors; then 2) employs a transformer-based neural architecture to find the device location from the sequence of velocity estimates. We only use an IMU sensor, which is energy efficient and privacy-preserving compared to WiFi, cameras, and other data sources. Our approach is significantly faster and achieves competitive results even compared with state-of-the-art methods that require a floorplan and run 20 to 30 times slower. We share our code, model and data at https://sachini.github.io/niloc.

## 4. Deep Learning Based Speed Estimation for Constraining Strapdown Inertial Navigation on Smartphones
- paper: https://paperswithcode.com/paper/deep-learning-based-speed-estimation-for
- github: https://github.com/AaltoVision/deep-speed-constrained-ins

Strapdown inertial navigation systems are sensitive to the quality of the data provided by the accelerometer and gyroscope. Low-grade IMUs in handheld smart-devices pose a problem for inertial odometry on these devices. We propose a scheme for constraining the inertial odometry problem by complementing non-linear state estimation by a CNN-based deep-learning model for inferring the momentary speed based on a window of IMU samples. We show the feasibility of the model using a wide range of data from an iPhone, and present proof-of-concept results for how the model can be combined with an inertial navigation system for three-dimensional inertial navigation.

## 5. RINS-W: Robust Inertial Navigation System on Wheels
- paper: https://cs.paperswithcode.com/paper/rins-w-robust-inertial-navigation-system-on
- github: https://github.com/mbrossar/RINS-W
  
This paper proposes a real-time approach for long-term inertial navigation based only on an Inertial Measurement Unit (IMU) for self-localizing wheeled robots. The approach builds upon two components: 1) a robust detector that uses recurrent deep neural networks to dynamically detect a variety of situations of interest, such as zero velocity or no lateral slip; and 2) a state-of-the-art Kalman filter which incorporates this knowledge as pseudo-measurements for localization. Evaluations on a publicly available car dataset demonstrates that the proposed scheme may achieve a final precision of 20 m for a 21 km long trajectory of a vehicle driving for over an hour, equipped with an IMU of moderate precision (the gyro drift rate is 10 deg/h). To our knowledge, this is the first paper which combines sophisticated deep learning techniques with state-of-the-art filtering methods for pure inertial navigation on wheeled vehicles and as such opens up for novel data-driven inertial navigation techniques. Moreover, albeit taylored for IMU-only based localization, our method may be used as a component for self-localization of wheeled robots equipped with a more complete sensor suite.

## 6-1. Improving Foot-Mounted Inertial Navigation Through Real-Time Motion Classification  
- paper: https://cs.paperswithcode.com/paper/improving-foot-mounted-inertial-navigation
- github: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a foot-mounted, zero-velocity-aided inertial navigation system (INS) by varying estimator parameters based on a real-time classification of motion type. We train a support vector machine (SVM) classifier using inertial data recorded by a single foot-mounted sensor to differentiate between six motion types (walking, jogging, running, sprinting, crouch-walking, and ladder-climbing) and report mean test classification accuracy of over 90% on a dataset with five different subjects. From these motion types, we select two of the most common (walking and running), and describe a method to compute optimal zero-velocity detection parameters tailored to both a specific user and motion type by maximizing the detector F-score. By combining the motion classifier with a set of optimal detection parameters, we show how we can reduce INS position error during mixed walking and running motion. We evaluate our adaptive system on a total of 5.9 km of indoor pedestrian navigation performed by five different subjects moving along a 130 m path with surveyed ground truth markers.

## 6-2. LSTM-Based Zero-Velocity Detection for Robust Inertial Navigation
- paper: https://cs.paperswithcode.com/paper/lstm-based-zero-velocity-detection-for-robust
- github: https://github.com/utiasSTARS/pyshoe

We present a method to improve the accuracy of a zero-velocity-aided inertial navigation system (INS) by replacing the standard zero-velocity detector with a long short-term memory (LSTM) neural network. While existing threshold-based zero-velocity detectors are not robust to varying motion types, our learned model accurately detects stationary periods of the inertial measurement unit (IMU) despite changes in the motion of the user. Upon detection, zero-velocity pseudo-measurements are fused with a dead reckoning motion model in an extended Kalman filter (EKF). We demonstrate that our LSTM-based zero-velocity detector, used within a zero-velocity-aided INS, improves zero-velocity detection during human localization tasks. Consequently, localization accuracy is also improved. Our system is evaluated on more than 7.5 km of indoor pedestrian locomotion data, acquired from five different subjects. We show that 3D positioning error is reduced by over 34% compared to existing fixed-threshold zero-velocity detectors for walking, running, and stair climbing motions. Additionally, we demonstrate how our learned zero-velocity detector operates effectively during crawling and ladder climbing. Our system is calibration-free (no careful threshold-tuning is required) and operates consistently with differing users, IMU placements, and shoe types, while being compatible with any generic zero-velocity-aided INS.

## 6-3. Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial Navigation  
- paper: https://cs.paperswithcode.com/paper/robust-data-driven-zero-velocity-detection 
- github: https://github.com/utiasSTARS/pyshoe

We present two novel techniques for detecting zero-velocity events to improve foot-mounted inertial navigation. Our first technique augments a classical zero-velocity detector by incorporating a motion classifier that adaptively updates the detector's threshold parameter. Our second technique uses a long short-term memory (LSTM) recurrent neural network to classify zero-velocity events from raw inertial data, in contrast to the majority of zero-velocity detection methods that rely on basic statistical hypothesis testing. We demonstrate that both of our proposed detectors achieve higher accuracies than existing detectors for trajectories including walking, running, and stair-climbing motions. Additionally, we present a straightforward data augmentation method that is able to extend the LSTM-based model to different inertial sensors without the need to collect new training data.


## 7. DeepNav: GPS-Denied Navigation Using Low-Cost Inertial Sensors and Recurrent Neural Networks
- paper: https://paperswithcode.com/paper/gps-denied-navigation-using-low-cost-inertial
- github: https://github.com/majuid/DeepNav

Autonomous missions of drones require continuous and reliable estimates for the drone's attitude, velocity, and position. Traditionally, these states are estimated by applying Extended Kalman Filter (EKF) to Accelerometer, Gyroscope, Barometer, Magnetometer, and GPS measurements. When the GPS signal is lost, position and velocity estimates deteriorate quickly, especially when using low-cost inertial sensors. This paper proposes an estimation method that uses a Recurrent Neural Network (RNN) to allow reliable estimation of a drone's position and velocity in the absence of GPS signal. The RNN is trained on a public dataset collected using Pixhawk. This low-cost commercial autopilot logs the raw sensor measurements (network inputs) and corresponding EKF estimates (ground truth outputs). The dataset is comprised of 548 different flight logs with flight durations ranging from 4 to 32 minutes. For training, 465 flights are used, totaling 45 hours. The remaining 83 flights totaling 8 hours are held out for validation. Error in a single flight is taken to be the maximum absolute difference in 3D position (MPE) between the RNN predictions (without GPS) and the ground truth (EKF with GPS). On the validation set, the median MPE is 35 meters. MPE values as low as 2.7 meters in a 5-minutes flight could be achieved using the proposed method. The MPE in 90% of the validation flights is bounded below 166 meters. The network was experimentally tested and worked in real-time.


## 8. CTIN: Robust Contextual Transformer Network for Inertial Navigation
- paper: https://paperswithcode.com/paper/ctin-robust-contextual-transformer-network
- github: https://github.com/bingrao/ctin

Recently, data-driven inertial navigation approaches have demonstrated their capability of using well-trained neural networks to obtain accurate position estimates from inertial measurement units (IMU) measurements. In this paper, we propose a novel robust Contextual Transformer-based network for Inertial Navigation~(CTIN) to accurately predict velocity and trajectory. To this end, we first design a ResNet-based encoder enhanced by local and global multi-head self-attention to capture spatial contextual information from IMU measurements. Then we fuse these spatial representations with temporal knowledge by leveraging multi-head attention in the Transformer decoder. Finally, multi-task learning with uncertainty reduction is leveraged to improve learning efficiency and prediction accuracy of velocity and trajectory. Through extensive experiments over a wide range of inertial datasets~(e.g. RIDI, OxIOD, RoNIN, IDOL, and our own), CTIN is very robust and outperforms state-of-the-art models.

## 9. AI-IMU Dead-Reckoning
- paper: https://ieeexplore.ieee.org/document/9035481
- github: https://github.com/mbrossar/ai-imu-dr

In this paper, we propose a novel accurate method for dead-reckoning of wheeled vehicles based only on an Inertial Measurement Unit (IMU). In the context of intelligent vehicles, robust and accurate dead-reckoning based on the IMU may prove useful to correlate feeds from imaging sensors, to safely navigate through obstructions, or for safe emergency stops in the extreme case of exteroceptive sensors failure. The key components of the method are the Kalman filter and the use of deep neural networks to dynamically adapt the noise parameters of the filter. The method is tested on the KITTI odometry dataset, and our dead-reckoning inertial method based only on the IMU accurately estimates 3D position, velocity, orientation of the vehicle and self-calibrates the IMU biases. We achieve on average a 1.10% translational error and the algorithm competes with top-ranked methods which, by contrast, use LiDAR or stereo vision.

## 10. End-to-End Learning Framework for IMU-Based 6-DOF Odometry
- paper: https://www.mdpi.com/1424-8220/19/17/3777
- github: https://github.com/jpsml/6-DOF-Inertial-Odometry

This paper presents an end-to-end learning framework for performing 6-DOF odometry by using only inertial data obtained from a low-cost IMU. The proposed inertial odometry method allows leveraging inertial sensors that are widely available on mobile platforms for estimating their 3D trajectories. For this purpose, neural networks based on convolutional layers combined with a two-layer stacked bidirectional LSTM are explored from the following three aspects. First, two 6-DOF relative pose representations are investigated: one based on a vector in the spherical coordinate system, and the other based on both a translation vector and an unit quaternion. Second, the loss function in the network is designed with the combination of several 6-DOF pose distance metrics: mean squared error, translation mean absolute error, quaternion multiplicative error and quaternion inner product. Third, a multi-task learning framework is integrated to automatically balance the weights of multiple metrics. In the evaluation, qualitative and quantitative analyses were conducted with publicly-available inertial odometry datasets. The best combination of the relative pose representation and the loss function was the translation and quaternion together with the translation mean absolute error and quaternion multiplicative error, which obtained more accurate results with respect to state-of-the-art inertial odometry techniques.

## 11. Deep Learning based Pedestrian Inertial Navigation: Methods, Dataset and On-Device Inference
-paper: https://paperswithcode.com/paper/deep-learning-based-pedestrian-inertial

Modern inertial measurements units (IMUs) are small, cheap, energy efficient, and widely employed in smart devices and mobile robots. Exploiting inertial data for accurate and reliable pedestrian navigation supports is a key component for emerging Internet-of-Things applications and services. Recently, there has been a growing interest in applying deep neural networks (DNNs) to motion sensing and location estimation. However, the lack of sufficient labelled data for training and evaluating architecture benchmarks has limited the adoption of DNNs in IMU-based tasks. In this paper, we present and release the Oxford Inertial Odometry Dataset (OxIOD), a first-of-its-kind public dataset for deep learning based inertial navigation research, with fine-grained ground-truth on all sequences. Furthermore, to enable more efficient inference at the edge, we propose a novel lightweight framework to learn and reconstruct pedestrian trajectories from raw IMU data. Extensive experiments show the effectiveness of our dataset and methods in achieving accurate data-driven pedestrian inertial navigation on resource-constrained devices.

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

## 3. Iterated-INS: Iterative Path Reconstruction for Large-Scale Inertial Navigation on Smartphones
- paper: https://paperswithcode.com/paper/190600360
- github: https://aaltoml.github.io/iterated-INS (no code implementations found)

Modern smartphones have all the sensing capabilities required for accurate and robust navigation and tracking. In specific environments some data streams may be absent, less reliable, or flat out wrong. In particular, the GNSS signal can become flawed or silent inside buildings or in streets with tall buildings. In this application paper, we aim to advance the current state-of-the-art in motion estimation using inertial measurements in combination with partial GNSS data on standard smartphones. We show how iterative estimation methods help refine the positioning path estimates in retrospective use cases that can cover both fixed-interval and fixed-lag scenarios. We compare estimation results provided by global iterated Kalman filtering methods to those of a visual-inertial tracking scheme (Apple ARKit). The practical applicability is demonstrated on real-world use cases on empirical data acquired from both smartphones and tablet devices.

## 4. Constructive Equivariant Observer Design for Inertial Navigation
- paper: https://paperswithcode.com/paper/constructive-equivariant-observer-design-for-1
- github: no code implementations found

Inertial Navigation Systems (INS) are algorithms that fuse inertial measurements of angular velocity and specific acceleration with supplementary sensors including GNSS and magnetometers to estimate the position, velocity and attitude, or extended pose, of a vehicle. The industry-standard extended Kalman filter (EKF) does not come with strong stability or robustness guarantees and can be subject to catastrophic failure. This paper exploits a Lie group symmetry of the INS dynamics to propose the first nonlinear observer for INS with error dynamics that are almost-globally asymptotically and locally exponentially stable, independently of the chosen gains. The observer is aided only by a GNSS measurement of position. As expected, the convergence guarantee depends on persistence of excitation of the vehicle's specific acceleration in the inertial frame. Simulation results demonstrate the observer's performance and its ability to converge from extreme errors in the initial state estimates.

## 5. Nonlinear Estimation for Position-Aided Inertial Navigation Systems
- paper: https://paperswithcode.com/paper/nonlinear-estimation-for-position-aided
- github: no code implementations found

In this work we solve the position-aided 3D navigation problem using a nonlinear estimation scheme. More precisely, we propose a nonlinear observer to estimate the full state of the vehicle (position, velocity, orientation and gyro bias) from IMU and position measurements. The proposed observer does not introduce additional auxiliary states and is shown to guarantee semi-global exponential stability without any assumption on the acceleration of the vehicle. The performance of the observer is shown, through simulation, to overcome the state-of-the-art approach that assumes negligible accelerations.

## 6. Synchronous Observer Design for Inertial Navigation Systems with Almost-Global Convergence
- paper: https://paperswithcode.com/paper/synchronous-observer-design-for-inertial

An Inertial Navigation System (INS) is a system that integrates acceleration and angular velocity readings from an Inertial Measurement Unit (IMU), along with other sensors such as GNSS position, GNSS velocity, and magnetometer, to estimate the attitude, velocity, and position of a vehicle. This paper shows that the INS problem can be analysed using the automorphism group of the extended special Euclidean group: a group we term the extended similarity group. By exploiting this novel geometric framework, we propose an observer architecture with synchronous error dynamics; that is, the error is stationary if the observer correction terms are set to zero. In turn, this enables us to derive a modular, or plug-and-play, observer design for INS that allows different sensors to be added or removed depending on what is available in the vehicle sensor suite. We prove both almost-global asymptotic and local exponential stability of the error dynamics for the common scenario of at least IMU and GNSS position. To the authors' knowledge, this is the first non-linear observer design with almost global convergence guarantees or with plug-and-play modular capability. A simulation with extreme initial error demonstrates the almost-global robustness of the system. Real-world capability is demonstrated on data from a fixed-wing UAV, and the solution is compared to the state-of-the-art ArduPilot INS.

## 7. iNavFIter-M: Matrix Formulation of Functional Iteration for Inertial Navigation Computation
- paper: https://paperswithcode.com/paper/inavfiter-m-matrix-formulation-of-functional

The acquisition of attitude, velocity, and position is an essential task in the field of inertial navigation, achieved by integrating the measurements from inertial sensors. Recently, the ultra-precision inertial navigation computation has been tackled by the functional iteration approach (iNavFIter) that drives the non-commutativity errors almost to the computer truncation error level. This paper proposes a computationally efficient matrix formulation of the functional iteration approach, named the iNavFIter-M. The Chebyshev polynomial coefficients in two consecutive iterations are explicitly connected through the matrix formulation, in contrast to the implicit iterative relationship in the original iNavFIter. By so doing, it allows a straightforward algorithmic implementation and a number of matrix factors can be pre-calculated for more efficient computation. Numerical results demonstrate that the proposed iNavFIter-M algorithm is able to achieve the same high computation accuracy as the original iNavFIter does, at the computational cost comparable to the typical two-sample algorithm. The iNavFIter-M algorithm is also implemented on a FPGA board to demonstrate its potential in real time applications.

## 8. A Closed-form Solution for the Strapdown Inertial Navigation Initial Value Problem
- paper: https://paperswithcode.com/paper/a-closed-form-solution-for-the-strapdown

Strapdown inertial navigation systems (SINS) are ubiquitious in robotics and engineering since they can estimate a rigid body pose using onboard kinematic measurements without knowledge of the dynamics of the vehicle to which they are attached. While recent work has focused on the closed-form evolution of the estimation error for SINS, which is critical for Kalman filtering, the propagation of the kinematics has received less attention. Runge-Kutta integration approaches have been widely used to solve the initial value problem; however, we show that leveraging the special structure of the SINS problem and viewing it as a mixed-invariant vector field on a Lie group, yields a closed form solution. Our closed form solution is exact given fixed gyroscope and accelerometer measurements over a sampling period, and it is utilizes 12 times less floating point operations compared to a single integration step of a 4th order Runge-Kutta integrator. We believe the wide applicability of this work and the efficiency and accuracy gains warrant general adoption of this algorithm for SINS.

## 9. Nonlinear Deterministic Filter for Inertial Navigation and Bias Estimation with Guaranteed Performance
- paper: https://paperswithcode.com/paper/nonlinear-deterministic-filter-for-inertial

Unmanned vehicle navigation concerns estimating attitude, position, and linear velocity of the vehicle the six degrees of freedom (6 DoF). It has been known that the true navigation dynamics are highly nonlinear modeled on the Lie Group of 
. In this paper, a nonlinear filter for inertial navigation is proposed. The filter ensures systematic convergence of the error components starting from almost any initial condition. Also, the errors converge asymptotically to the origin. Experimental results validates the robustness of the proposed filter.

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

## 4. Support Vector Machine for Determining Euler Angles in an Inertial Navigation System
- paper: https://paperswithcode.com/paper/support-vector-machine-for-determining-euler

The paper discusses the improvement of the accuracy of an inertial navigation system created on the basis of MEMS sensors using machine learning (ML) methods. As input data for the classifier, we used infor-mation obtained from a developed laboratory setup with MEMS sensors on a sealed platform with the ability to adjust its tilt angles. To assess the effectiveness of the models, test curves were constructed with different values of the parameters of these models for each core in the case of a linear, polynomial radial basis function. The inverse regularization parameter was used as a parameter. The proposed algorithm based on MO has demonstrated its ability to correctly classify in the presence of noise typical for MEMS sensors, where good classification results were obtained when choosing the optimal values of hyperpa-rameters.

# Multi-IMU inertial naviagtion source
## 1. ARRAY-IN: Inertial Navigation Using an Inertial Sensor Array
- paper: https://paperswithcode.com/paper/inertial-navigation-using-an-inertial-sensor
- github: https://github.com/hcarlsso/array-in

We present a comprehensive framework for fusing measurements from multiple and generally placed accelerometers and gyroscopes to perform inertial navigation. Using the angular acceleration provided by the accelerometer array, we show that the numerical integration of the orientation can be done with second-order accuracy, which is more accurate compared to the traditional first-order accuracy that can be achieved when only using the gyroscopes. Since orientation errors are the most significant error source in inertial navigation, improving the orientation estimation reduces the overall navigation error. The practical performance benefit depends on prior knowledge of the inertial sensor array, and therefore we present four different state-space models using different underlying assumptions regarding the orientation modeling. The models are evaluated using a Lie Group Extended Kalman filter through simulations and real-world experiments. We also show how individual accelerometer biases are unobservable and can be replaced by a six-dimensional bias term whose dimension is fixed and independent of the number of accelerometers.

## 2. MINS: Efficient and Robust Multisensor-aided Inertial Navigation System
- paper: https://cs.paperswithcode.com/paper/mins-efficient-and-robust-multisensor-aided
- github: https://github.com/rpng/mins

Robust multisensor fusion of multi-modal measurements such as IMUs, wheel encoders, cameras, LiDARs, and GPS holds great potential due to its innate ability to improve resilience to sensor failures and measurement outliers, thereby enabling robust autonomy. To the best of our knowledge, this work is among the first to develop a consistent tightly-coupled Multisensor-aided Inertial Navigation System (MINS) that is capable of fusing the most common navigation sensors in an efficient filtering framework, by addressing the particular challenges of computational complexity, sensor asynchronicity, and intra-sensor calibration. In particular, we propose a consistent high-order on-manifold interpolation scheme to enable efficient asynchronous sensor fusion and state management strategy (i.e. dynamic cloning). The proposed dynamic cloning leverages motion-induced information to adaptively select interpolation orders to control computational complexity while minimizing trajectory representation errors. We perform online intrinsic and extrinsic (spatiotemporal) calibration of all onboard sensors to compensate for poor prior calibration and/or degraded calibration varying over time. Additionally, we develop an initialization method with only proprioceptive measurements of IMU and wheel encoders, instead of exteroceptive sensors, which is shown to be less affected by the environment and more robust in highly dynamic scenarios. We extensively validate the proposed MINS in simulations and large-scale challenging real-world datasets, outperforming the existing state-of-the-art methods, in terms of localization accuracy, consistency, and computation efficiency. We have also open-sourced our algorithm, simulator, and evaluation toolbox for the benefit of the community: https://github.com/rpng/mins.

## 3. Wheel-INS2: Multiple MEMS IMU-based Dead Reckoning System for Wheeled Robots with Evaluation of Different IMU Configurations
- paper: https://cs.paperswithcode.com/paper/wheel-ins2-multiple-mems-imu-based-dead
- github: https://github.com/i2Nav-WHU/Wheel-INS

A reliable self-contained navigation system is essential for autonomous vehicles. Based on our previous study on Wheel-INS \cite{niu2019}, a wheel-mounted inertial measurement unit (Wheel-IMU)-based dead reckoning (DR) system, in this paper, we propose a multiple IMUs-based DR solution for the wheeled robots. The IMUs are mounted at different places of the wheeled vehicles to acquire various dynamic information. In particular, at least one IMU has to be mounted at the wheel to measure the wheel velocity and take advantages of the rotation modulation. The system is implemented through a distributed extended Kalman filter structure where each subsystem (corresponding to each IMU) retains and updates its own states separately. The relative position constraints between the multiple IMUs are exploited to further limit the error drift and improve the system robustness. Particularly, we present the DR systems using dual Wheel-IMUs, one Wheel-IMU plus one vehicle body-mounted IMU (Body-IMU), and dual Wheel-IMUs plus one Body-IMU as examples for analysis and comparison. Field tests illustrate that the proposed multi-IMU DR system outperforms the single Wheel-INS in terms of both positioning and heading accuracy. By comparing with the centralized filter, the proposed distributed filter shows unimportant accuracy degradation while holds significant computation efficiency. Moreover, among the three multi-IMU configurations, the one Body-IMU plus one Wheel-IMU design obtains the minimum drift rate. The position drift rates of the three configurations are 0.82\% (dual Wheel-IMUs), 0.69\% (one Body-IMU plus one Wheel-IMU), and 0.73\% (dual Wheel-IMUs plus one Body-IMU), respectively.

# Classic inertial navigation source
## 1. grizz_ins
 - github: https://github.com/thetengda/grizz_ins

Pure Ins integrator, tested with high-precision dataset

## 2. IC-GVINS: Exploring the Accuracy Potential of IMU Preintegration in Factor Graph Optimization
- paper: https://cs.paperswithcode.com/paper/exploring-the-accuracy-potential-of-imu
- github: https://github.com/i2nav-whu/ic-gvins

Inertial measurement unit (IMU) preintegration is widely used in factor graph optimization (FGO); e.g., in visual-inertial navigation system and global navigation satellite system/inertial navigation system (GNSS/INS) integration. However, most existing IMU preintegration models ignore the Earth's rotation and lack delicate integration processes, and these limitations severely degrade the INS accuracy. In this study, we construct a refined IMU preintegration model that incorporates the Earth's rotation, and analytically compute the covariance and Jacobian matrix. To mitigate the impact caused by sensors other than IMU in the evaluation system, FGO-based GNSS/INS integration is adopted to quantitatively evaluate the accuracy of the refined preintegration. Compared to a classic filtering-based GNSS/INS integration baseline, the employed FGO-based integration using the refined preintegration yields the same accuracy. In contrast, the existing rough preintegration yields significant accuracy degradation. The performance difference between the refined and rough preintegration models can exceed 200% for an industrial-grade MEMS module and 10% for a consumer-grade MEMS chip. Clearly, the Earth's rotation is the major factor to be considered in IMU preintegration in order to maintain the IMU precision, even for a consumer-grade IMU.

# Magnetic field navigation related source
## 1. ros2_magslam
 - github: https://github.com/JasonNg91/ros2_magslam

These packages can be used to perform mapping of the magnetic field using Gaussian Processes in ROS2 in psuedo real-time. 

## 2. geomag
- github: https://github.com/cmweiss/geomag/tree/master/geomag

Magnetic variation/declination: Calculates magnetic variation/declination for any latitude/longitude/altitude, for any date. Uses the NOAA National Geophysical Data Center, epoch 2015 data.

## 3. Dynamic Sensor Matching based on Geomagnetic Inertial Navigation
- paper: https://paperswithcode.com/paper/dynamic-sensor-matching-based-on-geomagnetic

Optical sensors can capture dynamic environments and derive depth information in near real-time. The quality of these digital reconstructions is determined by factors like illumination, surface and texture conditions, sensing speed and other sensor characteristics as well as the sensor-object relations. Improvements can be obtained by using dynamically collected data from multiple sensors. However, matching the data from multiple sensors requires a shared world coordinate system. We present a concept for transferring multi-sensor data into a commonly referenced world coordinate system: the earth's magnetic field. The steady presence of our planetary magnetic field provides a reliable world coordinate system, which can serve as a reference for a position-defined reconstruction of dynamic environments. Our approach is evaluated using magnetic field sensors of the ZED 2 stereo camera from Stereolabs, which provides orientation relative to the North Pole similar to a compass. With the help of inertial measurement unit informations, each camera's position data can be transferred into the unified world coordinate system. Our evaluation reveals the level of quality possible using the earth magnetic field and allows a basis for dynamic and real-time-based applications of optical multi-sensors for environment detection.

# IMU simulator related source
## 1. gnss-ins-simulator-from-Aceinna
- github: https://github.com/JasonNg91/gnss-ins-simulator-from-Aceinna

GNSS-INS-SIM is an GNSS/INS simulation project, which generates reference trajectories, IMU sensor output, GPS output, odometer output and magnetometer output. Users choose/set up the sensor model, define the waypoints and provide algorithms, and gnss-ins-sim can generate required data for the algorithms, run the algorithms, plot simulation results, save simulations results, and generate a brief summary.

## 2. IMU-Simulator opensource code from xioTechnologies
- github: https://github.com/JasonNg91/IMU-Simulator-from-xioTechnologies


# Human activity recognition source 
## 1. Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable Sensors
- paper: https://paperswithcode.com/paper/deep-residual-bidir-lstm-for-human-activity
- github: (1) https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition \
(2) https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs

Human activity recognition (HAR) has become a popular topic in research because of its wide application. With the development of deep learning, new ideas have appeared to address HAR problems. Here, a deep network architecture using residual bidirectional long short-term memory (LSTM) cells is proposed. The advantages of the new network include that a bidirectional connection can concatenate the positive time direction (forward state) and the negative time direction (backward state). Second, residual connections between stacked cells act as highways for gradients, which can pass underlying information directly to the upper layer, effectively avoiding the gradient vanishing problem. Generally, the proposed network shows improvements on both the temporal (using bidirectional cells) and the spatial (residual connections stacked deeply) dimensions, aiming to enhance the recognition rate. When tested with the Opportunity data set and the public domain UCI data set, the accuracy was increased by 4.78% and 3.68%, respectively, compared with previously reported results. Finally, the confusion matrix of the public domain UCI data set was analyzed.

## 2. Human Activity Recognition from Wearable Sensor Data Using Self-Attention
- paper: https://paperswithcode.com/paper/human-activity-recognition-from-wearable
- github: https://github.com/saif-mahmud/self-attention-HAR

Human Activity Recognition from body-worn sensor data poses an inherent challenge in capturing spatial and temporal dependencies of time-series signals. In this regard, the existing recurrent or convolutional or their hybrid models for activity recognition struggle to capture spatio-temporal context from the feature space of sensor reading sequence. To address this complex problem, we propose a self-attention based neural network model that foregoes recurrent architectures and utilizes different types of attention mechanisms to generate higher dimensional feature representation used for classification. We performed extensive experiments on four popular publicly available HAR datasets: PAMAP2, Opportunity, Skoda and USC-HAD. Our model achieve significant performance improvement over recent state-of-the-art models in both benchmark test subjects and Leave-one-subject-out evaluation. We also observe that the sensor attention maps produced by our model is able capture the importance of the modality and placement of the sensors in predicting the different activity classes.

## 3. Sequential Weakly Labeled Multi-Activity Localization and Recognition on Wearable Sensors using Recurrent Attention Networks
- paper: https://paperswithcode.com/paper/sequential-weakly-labeled-multi-activity
- github: https://github.com/KennCoder7/RAN

With the popularity and development of the wearable devices such as smartphones, human activity recognition (HAR) based on sensors has become as a key research area in human computer interaction and ubiquitous computing. The emergence of deep learning leads to a recent shift in the research of HAR, which requires massive strictly labeled data. In comparison with video data, activity data recorded from accelerometer or gyroscope is often more difficult to interpret and segment. Recently, several attention mechanisms are proposed to handle the weakly labeled human activity data, which do not require accurate data annotation. However, these attention-based models can only handle the weakly labeled dataset whose sample includes one target activity, as a result it limits efficiency and practicality. In the paper, we propose a recurrent attention networks (RAN) to handle sequential weakly labeled multi-activity recognition and location tasks. The model can repeatedly perform steps of attention on multiple activities of one sample and each step is corresponding to the current focused activity. The effectiveness of the RAN model is validated on a collected sequential weakly labeled multi-activity dataset and the other two public datasets. The experiment results show that our RAN model can simultaneously infer multi-activity types from the coarse-grained sequential weak labels and determine specific locations of every target activity with only knowledge of which types of activities contained in the long sequence. It will greatly reduce the burden of manual labeling. The code of our work is available at https://github.com/KennCoder7/RAN.

## 4. A benchmark of data stream classification for human activity recognition on connected objects
- paper: https://paperswithcode.com/paper/a-benchmark-of-data-stream-classification-for
- github: https://github.com/azazel7/paper-benchmark

This paper evaluates data stream classifiers from the perspective of connected devices, focusing on the use case of HAR. We measure both classification performance and resource consumption (runtime, memory, and power) of five usual stream classification algorithms, implemented in a consistent library, and applied to two real human activity datasets and to three synthetic datasets. Regarding classification performance, results show an overall superiority of the HT, the MF, and the NB classifiers over the FNN and the Micro Cluster Nearest Neighbor (MCNN) classifiers on 4 datasets out of 6, including the real ones. In addition, the HT, and to some extent MCNN, are the only classifiers that can recover from a concept drift. Overall, the three leading classifiers still perform substantially lower than an offline classifier on the real datasets. Regarding resource consumption, the HT and the MF are the most memory intensive and have the longest runtime, however, no difference in power consumption is found between classifiers. We conclude that stream learning for HAR on connected objects is challenged by two factors which could lead to interesting future work: a high memory consumption and low F1 scores overall.

## 5. B-HAR: an open-source baseline framework for in depth study of human activity recognition datasets and workflows
- paper: https://paperswithcode.com/paper/b-har-an-open-source-baseline-framework-for
- github: https://github.com/B-HAR-HumanActivityRecognition/B-HAR

Human Activity Recognition (HAR), based on machine and deep learning algorithms is considered one of the most promising technologies to monitor professional and daily life activities for different categories of people (e.g., athletes, elderly, kids, employers) in order to provide a variety of services related, for example to well-being, empowering of technical performances, prevention of risky situation, and educational purposes. However, the analysis of the effectiveness and the efficiency of HAR methodologies suffers from the lack of a standard workflow, which might represent the baseline for the estimation of the quality of the developed pattern recognition models. This makes the comparison among different approaches a challenging task. In addition, researchers can make mistakes that, when not detected, definitely affect the achieved results. To mitigate such issues, this paper proposes an open-source automatic and highly configurable framework, named B-HAR, for the definition, standardization, and development of a baseline framework in order to evaluate and compare HAR methodologies. It implements the most popular data processing methods for data preparation and the most commonly used machine and deep learning pattern recognition models.

## 6. Transformer Networks for Data Augmentation of Human Physical Activity Recognition
- paper: https://paperswithcode.com/paper/transformer-networks-for-data-augmentation-of
- github: https://github.com/sandeep-189/data-augmentation

Data augmentation is a widely used technique in classification to increase data used in training. It improves generalization and reduces amount of annotated human activity data needed for training which reduces labour and time needed with the dataset. Sensor time-series data, unlike images, cannot be augmented by computationally simple transformation algorithms. State of the art models like Recurrent Generative Adversarial Networks (RGAN) are used to generate realistic synthetic data. In this paper, transformer based generative adversarial networks which have global attention on data, are compared on PAMAP2 and Real World Human Activity Recognition data sets with RGAN. The newer approach provides improvements in time and savings in computational resources needed for data augmentation than previous approach.


# Open datasets 
1- RONIN which is available at here: https://ronin.cs.sfu.ca/

2- RIDI which is available at DropBox: https://www.dropbox.com/s/9zzaj3h3u4bta23/ridi_data_publish_v2.zip?dl=0

3- OxIOD: The Dataset for Deep Inertial Odometry which is available at OxIOD: http://deepio.cs.ox.ac.uk/

4- Px4 which can be downloaded from px4 and the scripts provided here has been used to download the data and pre-process it: https://github.com/majuid/DeepNav
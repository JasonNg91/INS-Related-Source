
## Notes of paper：Deep Learning for Inertial Positioning: A Survey  

- paper link: https://arxiv.org/abs/2303.03757
- Acknowledgment: Thanks to the paper author (Changhao Chen)'s awesome work!

- `综述范围：`narrow the survey coverage into deep learning based inertial positioning, so that offer deeper insights and analysis over the fast-evolving developments of this area in recent five years (2018-2022).
- `涵盖研究方向：`provide a comprehensive review on deep learning based inertial positioning approaches, from measurement calibration, inertial positioning algorithms to sensor fusion.

### Survey1: measurement calibration

![alt text](image.png)
![alt text](image-1.png)
`Deeplearning IMU校正的三种思路, 输入均为raw IMU数据：`

- (1) 模型输出：训练后的IMU数据；
- (2) 模型输出：训练后的姿态数据；
- (3) 模型输出：训练后的最优的典型校正算法的模型参数；

`风险/问题：`
- a.模型训练学习跟传感器、车辆会相关，泛化能力可能有问题？
- b.没有深入分析消除了哪部分噪声；

#### 1. 2019-A Survey of the Research Status of Pedestrian Dead Reckoning Systems Based on Inertial Sensors

- `主要贡献：采用1层ANN网络估计陀螺漂移`proposes to use 1-layer artificial neural network (ANN) to model the distribution of gyro drifts, and successfully approximate gyro drifts with such ’shallow’ network.
- `优势：`Compared with Kalman filtering (KF) based calibration method, its advantage is that it does not require to set hyber-parameters before use, such as the sensor noise matrix in KF.
- `不足：`N/A
  
#### 2. 2018-Improving inertial sensor by reducing errors using deep learning methodology

- `主要贡献：采用CNN网络估计低精度IMU的误差`presents a CNN (ConvNet) frmework to remove the error noises above inertial measurements.
- `优势：`deep learning can remove part of sensor error and improve test accuracy.
- `不足：(1)未在真实导航场景验证；（2）训练需要高精度IMU做真值参考；`this work is not validated in a real navigation setup, and thus can not show how the learning based sensor calibration reduces error drifts of inertial navigation.


#### 3. 2019-Orinet: Robust 3-d orientation estimation with a single particular imu.

- `主要贡献：`OriNet inputs 3-dimensional gyroscope signals into a long-short-termmemory (LSTM) network to obtain calibrated gyroscope signals, that are integrated with the orientation at the previous timestep to generate orientation estimates at current current timestep. 
- `优势：直接估计更精准的姿态，参考真值为姿态。`A loss function between orientation estimates and
real orientation is defined and minimized for model training.
- `不足：`N/A

#### 4. 2020-Denoising imu gyroscopes with deep learning for open-loop attitude estimation

- `主要贡献：`learn gyro corrections to calibrate gyroscope
- `优势：`learns to calibrate gyroscope but using convolutional neural networks (ConvNet) instead, reporting good attitude estimation accuracy as well.
- `不足：`N/A

#### 5. 2022-Calib-net: Calibrating the low-cost imu via deep convolutional neural network.

- `主要贡献：`To extract effective spatiotemporal features from inertial data, Calib-Net is based on
dilation ConvNet to compensate the gyro noise.
- `优势：`this model is able to reduce orientation error largely, compared with raw IMU integration. When this learned inertial calibration model is incorporated into a visual-inertial odometry (VIO), it further improves localization performance, and outperforms representative VIO
- `不足：`N/A

#### 6. 2022-A mems imu gyroscope calibration method based on deep learning.

#### 7. 2020-Learning to compensate for the drift and error of gyroscope in vehicle localization.

#### 8. 2019-Learning to calibrate: Reinforcement learning for guided calibration of visual–inertial rigs.

- `主要贡献：`models inertial sensor calibration as a Markov Decision Process, and propose to learn the optimal calibration parameters via deep reinforcement learning
- `优势：学习IMU典型校正算法的模型参数，然后用于传统校正算法`evaluated their method in successfully calibrating inertial sensor for a VIO system.
- `不足：`N/A

### Survey2: inertial positioning

otes of paper：Deep Learning for Inertial Positioning: A Survey  

- paper link: https://arxiv.org/abs/2303.03757
- Acknowledgment: Thanks to the paper author (Changhao Chen)'s awesome work!

- `综述范围：`narrow the survey coverage into deep learning based inertial positioning, so that offer deeper insights and analysis over the fast-evolving developments of this area in recent five years (2018-2022).
- `涵盖研究方向：`provide a comprehensive review on deep learning based inertial positioning approaches, from measurement calibration, inertial positioning algorithms to sensor fusion.

### Survey1: measurement calibration

![alt text](image.png)
![alt text](image-1.png)
`Deeplearning IMU校正的三种思路, 输入均为raw IMU数据：`

- (1) 模型输出：训练后的IMU数据；
- (2) 模型输出：训练后的姿态数据；
- (3) 模型输出：训练后的最优的典型校正算法的模型参数；

`风险/问题：`
- a.模型训练学习跟传感器、车辆会相关，泛化能力可能有问题？
- b.没有深入分析消除了哪部分噪声；

#### 1. 2019-A Survey of the Research Status of Pedestrian Dead Reckoning Systems Based on Inertial Sensors

- `主要贡献：采用1层ANN网络估计陀螺漂移`proposes to use 1-layer artificial neural network (ANN) to model the distribution of gyro drifts, and successfully approximate gyro drifts with such ’shallow’ network.
- `优势：`Compared with Kalman filtering (KF) based calibration method, its advantage is that it does not require to set hyber-parameters before use, such as the sensor noise matrix in KF.
- `不足：`N/A
  
#### 2. 2018-Improving inertial sensor by reducing errors using deep learning methodology

- `主要贡献：采用CNN网络估计低精度IMU的误差`presents a CNN (ConvNet) frmework to remove the error noises above inertial measurements.
- `优势：`deep learning can remove part of sensor error and improve test accuracy.
- `不足：(1)未在真实导航场景验证；（2）训练需要高精度IMU做真值参考；`this work is not validated in a real navigation setup, and thus can not show how the learning based sensor calibration reduces error drifts of inertial navigation.


#### 3. 2019-Orinet: Robust 3-d orientation estimation with a single particular imu.

- `主要贡献：`OriNet inputs 3-dimensional gyroscope signals into a long-short-termmemory (LSTM) network to obtain calibrated gyroscope signals, that are integrated with the orientation at the previous timestep to generate orientation estimates at current current timestep. 
- `优势：直接估计更精准的姿态，参考真值为姿态。`A loss function between orientation estimates and
real orientation is defined and minimized for model training.
- `不足：`N/A

#### 4. 2020-Denoising imu gyroscopes with deep learning for open-loop attitude estimation

- `主要贡献：`learn gyro corrections to calibrate gyroscope
- `优势：`learns to calibrate gyroscope but using convolutional neural networks (ConvNet) instead, reporting good attitude estimation accuracy as well.
- `不足：`N/A

#### 5. 2022-Calib-net: Calibrating the low-cost imu via deep convolutional neural network.

- `主要贡献：`To extract effective spatiotemporal features from inertial data, Calib-Net is based on
dilation ConvNet to compensate the gyro noise.
- `优势：`this model is able to reduce orientation error largely, compared with raw IMU integration. When this learned inertial calibration model is incorporated into a visual-inertial odometry (VIO), it further improves localization performance, and outperforms representative VIO
- `不足：`N/A

#### 6. 2022-A mems imu gyroscope calibration method based on deep learning.

#### 7. 2020-Learning to compensate for the drift and error of gyroscope in vehicle localization.

#### 8. 2019-Learning to calibrate: Reinforcement learning for guided calibration of visual–inertial rigs.

- `主要贡献：`models inertial sensor calibration as a Markov Decision Process, and propose to learn the optimal calibration parameters via deep reinforcement learning
- `优势：学习IMU典型校正算法的模型参数，然后用于传统校正算法`evaluated their method in successfully calibrating inertial sensor for a VIO system.
- `不足：`N/A

### Survey2: inertial positioning

![alt text](image-2.png)
![alt text](image-3.png)
`Deeplearning 惯性递推, 输入均为raw IMU数据：`

- (1) 模型输出：训练后的2D/3D位移变化量和uncentrainty/covariance；
- (2) 模型输出：训练后的速度估计值；
- (3) 模型输出：训练后的最优的典型校正算法的模型参数；
#### 1. 2018-Ionet: Learning to cure the curse of drift in inertial odometry. (Changhao Chen)

- `主要贡献：估计2D位移变化量`They propose IONet, an LSTM based framework for end-to-end learning of relative
poses. They formulate inertial positioning as a sequential learning problem with a key observation that 2D motion displacements in the polar coordinate.
- `优势：`the frequency of platform vibrations is relevant to absolute moving speed, that can be measured by IMU, when tracking human or wheeled configurations. To train neural models, a large collection of data was collected from a smartphone based IMU in a room with high-precision visual motion tracking system (i.e. Vicon) to provide ground-truth pose labels.
- `不足：`依赖数据集，没有训练到的数据性能会下降；

#### 2. 2019-Motiontransformer: Transferring neural inertial tracking between domains. (Changhao Chen)

- `主要贡献：生成式学习，新场景自适应，提升泛化能力`proposes MotionTransformer, allowing inertial positioning model to self-adapt into new domains via deep generative models and domain adaptation technique
- `优势：`without the need of labels in new domains.
- `不足：`N/A

#### 3. 2021-Deep neural network based inertial odometry using low-cost inertial measurement units. (Changhao Chen)

- `主要贡献：神经网络估计位姿的同时输出uncentrainty`produce pose uncertainties along with poses, offering the belief in to what extent the learned pose can be trusted.
- `优势：`神经网络估计位姿的同时输出uncentrainty
- `不足：`N/A

#### 4. 2020-Tlio: Tight learned inertial odometry

- `主要贡献：估计3D位移变化量和Covariance，并与EKF结合`proposes to learn 3D location displacements and covariances from a sequence of gravity aligned inertial data. To avoid the impacts from initial orientation, the inertial data are transformed into a local gravity-aligned frame
- `优势：`The learned 3D displacements and covariances are then incorporated into an extended Kalman filter as observation states that estimates fullstates of orientation, velocity, location and IMU bias.
- `不足：`RIDI, RoNIN and TLIO still require device orientation to rotate inertial data into a frame.

#### 5. 2018-Ridi: Robust imu double integration

- `主要贡献：估计出速度向量，修正加速度测量值，然后双重积分得到位置`RIDI trains a deep neural network to regress velocity vectors from inertial data, which are then used to correct linear accelerations (i.e. acceleration measurements minus gravity).Finally, these corrected linear accelerations are doubly integrated into positions.
- `优势：`Learning human walking speed is a good prior to improve corrupted inertial accelerations, so that the unconstrained drifts of inertial positioning are correspondingly
compensated and restricted into a lower level.
- `不足：`RIDI, RoNIN and TLIO still require device orientation to rotate inertial data into a frame.

#### 6. 2020-Ronin: Robust neural inertial navigation in the wild: Benchmark, evaluations, & new methods

- `主要贡献：相比RIDI进一步做坐标变换，减少对绝对航向的依赖`RoNIN improves RIDI by transforming inertial measurements and learned velocity vector in heading-agnostic coordinate frame and introducing several novel velocity losses.
- `优势：`To reduce the influences from orientation, RoNIN uses device orientation to
transform inertial data in a frame whose Z axis is aligned with gravity.
- `不足：`so that one limitation is its reliance on orientation estimation.

#### 7. 2022-Neural inertial localization (需要再理解下)

- `主要贡献：`Based on RoNIN, an interesting trial is NILoc, to solve the so-called neural inertial localization problem, inferring global location from inertial motion history only. This work observes that human experiences unique motion at different locations, so that these motion patterns can be exploited as ”fingerprinting” to determine the location, similar to WiFi or magnetic-field fingerprinting.
- `优势：`NILoc first calculates a sequence of velocity from inertial data, and then adopts
Transformer based DNN framework to transform velocity sequence to location.
- `不足：`one fundamental limitation is that in some areas there is no unique motion pattern, e.g. open space, symmetry or repetitive place.

#### 8. 2018-Deep learning based speed estimation for constraining strapdown inertial navigation on smartphones. 

- `主要贡献：利用神经网络估计速度作为KF量测+SINS结合`leverages ConvNet to infer current speed from IMU sequence, and uses this speed into Kalman filtering as velocity observation to constrain the drifts of SINS based inertial positioning.
- `优势：`It is similar to ZUPT method, detecting and using zero-velocity into KF as observations, but instead it uses full-speeds as observations in KF. Using learned velocity extends the usage of KF to more complex human motion.
- `不足：`N/A

#### 9. 2020-Pedestrian motion tracking by using inertial sensors on the smartphone.

- `主要贡献：学习行人速度，输出噪声参数动态调整传统KF参数` based on DNN to infer walking velocity in the body frame, and combines it with an extended KF. Except the learned velocity, it learns to produce noise parameter for KF to dynamically update parameters instead of setting a fixed noise parameter.
- `优势：`动态调整KF参数
- `不足：`N/A

#### 10. 2022-Ctin: Robust contextual transformer network for inertial navigation 

- `主要贡献：contextual transformer network`CTIN introduces attention module into DNN based inertial odometry model to aggregate the local and global information of input data.
- `优势：`带细看研究
- `不足：`N/A

#### 11. 2022-A2dio: Attention-driven deep inertial odometry for pedestrian localization based on 6d imu

- `主要贡献：attention mechanism`A2DIO uses attention mechanism above LSTM into learning framework.
- `优势：`带细看研究
- `不足：`N/A

#### 12. 2021-Robust inertial motion tracking through deep sensor fusion across smart earbuds and smartphone.

- `主要贡献：手机+耳机双设备IMU协同，更鲁棒`DeepIT fuses inertial information from two mobile devices, i.e., a smartphone and an earbud, showing that their multi-sensors work is more robust to challenging unconstrained human motion than single IMU based approach.
- `优势：`更鲁棒
- `不足：`N/A

#### 13. 2021-Idol: Inertial deep orientation estimation and localization.

- `主要贡献：不依赖设备姿态，自己分两步估计，先估计姿态，再估计位置`To mitigate the reliance on device orientation, IDOL proposes a two-stages process, which first learns orientation from data, rotates inertial data into suitable frame, and finally learns position
- `优势：`不依赖设备姿态估计，靠自身网络进行估计
- `不足：`N/A

#### 14. 2022-Magnetic field-enhanced learning-based inertial odometry for indoor pedestrian

- `主要贡献：利用磁力计估计航向，与神经网络里程计结合`uses magnetic data to estimate orientation, being combined with learned odometry to reduce positioning drifts.
- `优势：`
- `不足：`N/A

### Survey3: sensor fusion

#### . 

- `主要贡献：`
- `优势：`
- `不足：`N/A
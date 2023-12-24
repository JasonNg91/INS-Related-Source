# INS-Related-Source
Inertial navigation related source

# AI-based inertial navigation source
## 1. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods
paper:https://paperswithcode.com/paper/ronin-robust-neural-inertial-navigation-in
github: 
(1) https://github.com/Sachini/ronin
(2) https://github.com/BehnamZeinali/IMUNet

RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods
30 May 2019  ·  Hang Yan, Sachini Herath, Yasutaka Furukawa
This paper sets a new foundation for data-driven inertial navigation research, where the task is the estimation of positions and orientations of a moving subject from a sequence of IMU sensor measurements. More concretely, the paper presents 1) a new benchmark containing more than 40 hours of IMU sensor data from 100 human subjects with ground-truth 3D trajectories under natural human motions; 2) novel neural inertial navigation architectures, making significant improvements for challenging motion cases; and 3) qualitative and quantitative evaluations of the competing methods over three inertial navigation benchmarks. We will share the code and data to promote further research.

## 2. NILoc:Neural Inertial Localization
paper: https://openaccess.thecvf.com/content/CVPR2022/html/Herath_Neural_Inertial_Localization_CVPR_2022_paper.html
github:https://github.com/Sachini/niloc
This paper proposes the inertial localization problem, the task of estimating the absolute location from a sequence of inertial sensor measurements. This is an exciting and unexplored area of indoor localization research, where we present a rich dataset with 53 hours of inertial sensor data and the associated ground truth locations. We developed a solution, dubbed neural inertial localization (NILoc) which 1) uses a neural inertial navigation technique to turn inertial sensor history to a sequence of velocity vectors; then 2) employs a transformer-based neural architecture to find the device location from the sequence of velocity estimates. We only use an IMU sensor, which is energy efficient and privacy-preserving compared to WiFi, cameras, and other data sources. Our approach is significantly faster and achieves competitive results even compared with state-of-the-art methods that require a floorplan and run 20 to 30 times slower. We share our code, model and data at https://sachini.github.io/niloc.

# Classic inertial navigation source

# IMU simulator related source

# Open datasets 

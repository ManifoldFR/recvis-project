# Project: _Humanoid Robot Imitation of Human Motion from Instructional Videos_

Project for the MVA "Object Recognition and Computer Vision" class (https://www.di.ens.fr/willow/teaching/recvis19/).

Clone the repository
 ```bash
git clone https://github.com/ManifoldFR/recvis-project
 ```


## Dependencies

The initial [HMR repository](https://github.com/akanazawa/hmr) uses Python 2.7 and TensorFlow 1.3. We patched HMR to be able to use more recent versions of Python and TensorFlow 1.x (tested on TensorFlow 1.15 and Python 3.7).

We reverse-engineered [motion-reconstruction](https://github.com/akanazawa/motion_reconstruction) to use with [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch) (included in the repository) and Python 3 instead of Openpose.

The patched versions of the repos are included in this repository.


DeepMimic requires a specific format for motion capture files. They can be obtained from BVH (BioVision Hierarchy Animation) files. Converting from BVH to DeepMimic requires the [BvhToDeepMimic](https://github.com/BartMoyaers/BvhToDeepMimic) package
```bash
pip install bvhtodeepmimic
```

## Use

Put the videos in a directory (by default `data/vid`) and call
```bash
python -m run_alphapose
```
Then run
```bash
python -m refine_hmr
```

### Conversion of motion to DeepMimic JSON format

Use https://github.com/BartMoyaers/BvhToDeepMimic to convert the BVH files to DeepMimic-formatted files.

We repurposed MoCap conversion files from the PyBullet reimplementation of DeepMimic (credit to Erwin Coumans and Yihang Yin), [inverse_kinematics.py](https://github.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/inverse_kinematics.py) and
[transformation.py](https://github.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py).
We wrote a [wrapper](./ik_hmr_deepmimic.py) for this code that you can modify and call as
```
python ik_hmr_deepmimic.py
```



## Bibliography

* SFV: Reinforcement Learning of Physical Skills from Videos, Peng, Kanazawa, Malik, Abbeel and Levine. In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2018). https://xbpeng.github.io/projects/SFV/2018_TOG_SFV.pdf | [GitHub project page](https://xbpeng.github.io/projects/SFV/index.html) [GitHub code repo](https://github.com/akanazawa/motion_reconstruction)
* DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills: https://xbpeng.github.io/projects/DeepMimic/index.html
* Estimating 3D Motion and Forces of Person-Object Interactionsfrom Monocular Video. https://arxiv.org/pdf/1904.02683.pdf | [Project page](https://www.di.ens.fr/willow/research/motionforcesfromvideo/research/li19mfv/)
* End-to-end Recovery of Human Shape and Pose: https://arxiv.org/pdf/1712.06584.pdf | [GitHub repo](https://github.com/akanazawa/hmr)
* RMPE: Regional Multi-person Pose Estimation: https://arxiv.org/abs/1612.00137.pdf | [Project page](https://www.mvig.org/research/alphapose.html)

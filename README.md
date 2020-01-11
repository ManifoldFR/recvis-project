# Project: _Humanoid Robot Imitation of Human Motion from Instructional Videos_

Clone the repository
 ```bash
git clone https://github.com/ManifoldFR/recvis-project
 ```


## Dependencies

The initial HMR repository uses Python 2.7 and TensorFlow 1.3. We patched HMR to be able to use more recent versions of Python and TensorFlow 1.x (tested on TensorFlow 1.15 and Python 3.7).

We reverse-engineered [motion-reconstruction](https://github.com/akanazawa/motion_reconstruction) to use with AlphaPose (included in the repository) and Python 3 instead of Openpose.

## Use

Put the videos in a directory (by default `data/vid`) and call
```bash
python -m run_alphapose
```
Then run
```bash
python -m refine_hmr.py
```

## Bibliography

* SFV: Reinforcement Learning of Physical Skills from Videos, Peng, Kanazawa, Malik, Abbeel and Levine. In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2018). https://xbpeng.github.io/projects/SFV/2018_TOG_SFV.pdf | [GitHub project page](https://xbpeng.github.io/projects/SFV/index.html) [GitHub code repo](https://github.com/akanazawa/motion_reconstruction)
* DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills: https://xbpeng.github.io/projects/DeepMimic/index.html
* Estimating 3D Motion and Forces of Person-Object Interactionsfrom Monocular Video. https://arxiv.org/pdf/1904.02683.pdf | [Project page](https://www.di.ens.fr/willow/research/motionforcesfromvideo/research/li19mfv/)
* End-to-end Recovery of Human Shape and Pose: https://arxiv.org/pdf/1712.06584.pdf | [GitHub repo](https://github.com/akanazawa/hmr)
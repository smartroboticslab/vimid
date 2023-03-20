# VI-MID

## [Project Page](https://mlr.in.tum.de/research/projects/vimid) | [Video](https://youtu.be/6GY5cBwvuJE) | [Paper](https://arxiv.org/pdf/2208.04274.pdf)


This repository contains VI-MID: a new multi-instance dynamic RGBD-Inertial SLAM system using an object-level octree-based volumetric representation.

![VI-MID](teaser/VI-MID.gif)


We present a tightly-coupled visual-inertial object-level multi-instance dynamic SLAM system. Even in extremely dynamic scenes, it can robustly optimise for the camera pose, velocity, IMU biases and build a dense 3D reconstruction object-level map of the environment. Our system can robustly track and reconstruct the geometries of arbitrary objects, their semantics and motion by incrementally fusing associated colour, depth, semantic, and foreground object probabilities into each object model thanks to its robust sensor and object tracking. In addition, when an object is lost or moved outside the camera field of view, our system can reliably recover its pose upon re-observation. We demonstrate the robustness and accuracy of our method by quantitatively and qualitatively testing it in real-world data sequences.


## TODO:

- [x] Create custom dataset
- [ ] Evaluation examples
- [x] Documentation
- [x] License

## How to use our software
### Dependencies
Check `Dockerfile` and `.devcontainer/docker-compose.yml` for the required dependencies. We provided a [vscode devcontainer](https://code.visualstudio.com/docs/devcontainers/create-dev-container) for easy development.
`xhost +` is required to run the container with GUI support.
If you are using remote computer for developmnt without a monitor and meet an issue to get GUI: try a hack `export DISPLAY=$LOCAL_DISPLAY$`

### Installation:
Get our code and submodules:
```
git clone git@github.com:binbin-xu/vimid.git
git submodule update --init --recursive
```
Install the custom OKVIS and realsense library in the `third-party` folder. 

Then go into the ***apps/kfusion*** folder, simply run the following command to build the software and the dependencies:

```
make
```
If any error occurs, please check the dockerfile for any possible dependencies.

### Uninstall:

```
make clean
```

<!-- ### Dependency
If you meet any dependency issue in compiling, please refer to [github action file](https://github.com/binbin-xu/mid-fusion/blob/master/.github/workflows/main.yml) or [report an issue](https://github.com/binbin-xu/mid-fusion/issues).
 -->

### Demo:

We provide some usage samples. Please see the bash files in the ***apps/kfusion/demo***  folder.

The data used to run those bash can be downloaded via [this link](https://drive.google.com/drive/folders/1Hn0M1assEKimjLWulQBTiWJARa3fdrUn?usp=sharing). Remember to modify the datasets address in the bash files accordingly.

### Customised settings:

RGB-D sequences need to be given as the demo format. Specifically, RGB images in ***/cam0*** folder, depth images in ***/depth0*** folder, mask-RCNN outputs in ***/mask_RCNN*** folder, IMU data in ***/imu0*** folder (***/cam1***, ***/cam0_ori*** can be ignored). If there's no alignment between RGB and depth image, you may need to associate them (check [this link](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) for details). \
The input images are defined as in the TUM RGBD datasets, where the input images are in the resolution of 640 X 480 and depth is scaled by 5000. Images are named in the recorded timestamples (nanoseconds).

You may need to tune some hyper-paramters defined in the [file](https://github.com/binbin-xu/vimid_alpha/blob/main/apps/kfusion/include/default_parameters.h) and parse them as arguments for your own sequences. \
Make sure to complie and run in debug mode first to expose bugs that were hidden in unoptimized code.
```
make debug
```

Then you can run our modified Mask RCNN script (check [demo/vimid-mask.py](https://github.com/binbin-xu/detectron2_for_vimid/blob/master/demo/vimid-mask.py) in [this repo](https://github.com/binbin-xu/detectron2_for_vimid)) to generate masks, classes, and semantic probability in cnpy format. Here we provide a [detectron2](https://github.com/binbin-xu/detectron2_for_vimid) version for usage. We did not finetune the pretrained coco models, and the results would be much improved with a better/more suited segmentation mask. Therefore if you want to increase performance in your specific domain, please **consider training a network on your data**. 

Those tunable parameters can be found in `apps/kfusion/include/default_parameters.h`.


<!-- ### Difference with supereight implementation:

  This is an official implementation of MID-Fusion system. The system is implemented based on [supereight](https://github.com/emanuelev/supereight), an octree-based volumetric representation. However, MID-Fusion was developed in parallel with supereight system. Therefore you may notice some structure and contents differences between the two systems. We are trying to merge the new updates from supereight into our MID-Fusion implementation. -->



## Citations

Please consider citing this project in your publications if it helps your work. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

```
@inproceedings{Ren:Xu:etal:IROS2022,
  title={Visual-Inertial Multi-Instance Dynamic SLAM with Object-level Relocalisation},
  author={Ren, Yifei and Xu, Binbin and Choi, Christopher L and Leutenegger, Stefan},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11055--11062},
  year={2022},
}

@inproceedings{Xu:etal:ICRA2019,
author = {Binbin Xu and Wenbin Li and Dimos Tzoumanikas and Michael Bloesch and Andrew Davison and Stefan Leutenegger},
booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
+ title = {{MID-Fusion}: Octree-based Object-Level Multi-Instance Dynamic SLAM},
year = {2019},
}
```


## License
Copyright © 2017-2023 Smart Robotics Lab, Imperial College London \
Copyright © 2021-2023 Yifei Ren \
Copyright © 2017-2023 Binbin Xu

Distributed under the [BSD 3-clause license](LICENSE).



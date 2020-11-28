# Hourglass for corner detection of deformable fabric

This repo contains the code structure used for the detection of the corners of a wrinkled towel. 

## Installation

**Create conda environment**
```
$ conda create -n ENVIRONMENT_NAME python=3
$ conda activate ENVIRONMENT_NAME
```

**Clone and install requirements**  
```
$ git clone git@github.com:AlbertMitjans/pytorch-corner-detection.git
$ cd pytorch-corner-detection/
$ conda install --file requirements.txt
```
**Download pretrained weights**
```
$ cd checkpoints/
$ bash get_weights.sh
```
**Download dataset**
```
$ cd data/
$ bash get_dataset.sh
```  

## Run test

Evaluates the model on the dataset.

```
$ python3 main.py --train False --ckpt checkpoints/best_ckpt/model.pth
```
As default, for every case, the depth, the output, and the rgb are saved in the *output/* folder.

<p align="center">
  <img width="900" height="350" src="assets/output.png">
</p>

**Testing log**
```
* Recall(%): 54.830     * Precision(%):  (97.727, 79.804, 41.202, 14.815)    
```

The precision is computed for the (1, 2, 3, 4) corners detected with highest confidence (the gaussians with a highest value on its center).

## Run train

Trains the network from scratch or from a given ckpt.

```
$ python3 main.py
```

**Training log**
```
Epoch: [5][300/312]	Loss.avg: 0.3615	Recall(%): 21.622	Precision num. corners (%): (22.591, 18.563, 15.833, 16.809)
```

**Tensorboard**

Track training progress in Tensorboard:
+ Initialize training
+ Run the command below inside the pytorch-corner-detection directory.
+ Go to [http://localhost:6006/](http://localhost:6006/)

```
$ tensorboard --logdir='logs' --port=6006
```


## Real-time display and image capture
1. Watch the output of the network in real-time by connecting a depth camera.

2. Capture images to increase the size of the dataset.

In order to work with the encoded images received by the camera, we need to convert them to OpenCV images using the *cv_bridge* package. In python 3, this package is not compatible with rospy, therefore I created [another repository](https://github.com/AlbertMitjans/real-time) which implements this in python 2.

<p align="center">
  <img width="800" height="300" src="assets/real-time.png">
</p>

## Arguments
--train (default:True) : if True/False, training/testing is implemented.  
--val_data (default:True) : if True/False, all/validation data will be evaluated.  
--save_imgs (default:True) : if True output images will be saved in the \Output folder.  
--batch_size (default:1)  
--depth (default:True) : if True/False, depth/RGB images will be used.  
--ckpt(default:None)  
--num_epochs (default:200)  



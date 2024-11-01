# Efficient Indoor 3D Object Detection in Point Clouds Using the Kinect Sensor


## Introduction
This paper was published in Measurement Science and Technology(DOI 10.1088/1361-6501/ad8d73)

Only using the point clouds collected by a Kinect sensor to perform 3D object detection is a very challenging task for indoor sence understanding. In this paper, we propose a novel point clouds based 3D object detection framework named Weighted VoteNet (W-VoteNet), which systematically enhances seed points, voting points, and proposal objects to varying degree, thereby enhancing the detection accuracy of indoor scene point clouds. Firstly, a new neighborhood feature enhancement module was designed to enhance the features of the seed point, which can better utilize the neighborhood feature of the seed point. Secondly, a weighted voting module is implemented to increase the accuracy of voting, allowing more prospect seed points to participate in voting. Finally, a new semantic relation reasoning module is proposed to get the semantic relationship features of the proposal object, which can further decrease false alarm rate. The three modules we proposed all contribute to more accurate voting and more effective proposals. The experimental results on two benchmark indoor 3D object detection datasets, SUN RGB-D and ScanNet V2, demonstrate the effectiveness of our approach. 

## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: After a code update on 2/6/2020, the code is now also compatible with Pytorch v1.2+

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd pointnet2
    python setup.py install

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Run demo

You can download pre-trained models and sample point clouds [HERE](https://drive.google.com/file/d/1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l/view?usp=sharing).
Unzip the file under the project root path (`/path/to/project/demo_files`) and then run:

    python demo.py

The demo uses a pre-trained model (on SUN RGB-D) to detect objects in a point cloud from an indoor room of a table and a few chairs (from SUN RGB-D val set). You can use 3D visualization software such as the [MeshLab](http://www.meshlab.net/) to open the dumped file under `demo_files/sunrgbd_results` to see the 3D detection output. Specifically, open `***_pc.ply` and `***_pred_confident_nms_bbox.ply` to see the input point cloud and predicted 3D bounding boxes.

You can also run the following command to use another pretrained model on a ScanNet:

    python demo.py --dataset scannet --num_point 40000

Detection results will be dumped to `demo_files/scannet_results`.

## Training and evaluating

### Data preparation

For SUN RGB-D, follow the [README](https://github.com/facebookresearch/votenet/blob/master/sunrgbd/README.md) under the `sunrgbd` folder.

For ScanNet, follow the [README](https://github.com/facebookresearch/votenet/blob/master/scannet/README.md) under the `scannet` folder.

### Train and test on SUN RGB-D

To train a new VoteNet model on SUN RGB-D data (depth images):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options (e.g. you can also set `--model boxnet` to train with the baseline BoxNet model).
While training you can check the `log_sunrgbd/log_train.txt` file on its progress, or use the TensorBoard to see loss curves.

To test the trained model with its checkpoint:

    python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_sunrgbd` folder (or any other folder you specify). You can run `python eval.py -h` to see the full options for evaluation. After the evaluation, you can use MeshLab to visualize the predicted votes and 3D bounding boxes (select wireframe mode to view the boxes).
Final evaluation results will be printed on screen and also written in the `log_eval.txt` file under the dump directory. In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes. A properly trained VoteNet should have around 57 mAP@0.25 and 32 mAP@0.5.

### Train and test on ScanNet

To train a VoteNet model on Scannet data (fused scan):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset scannet --log_dir log_scannet --num_point 40000

To test the trained model with its checkpoint:

    python eval.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on axis aligned boxes. A properly trained VoteNet should have around 58 mAP@0.25 and 35 mAP@0.5.

### Train on your own data

[For Pro Users] If you have your own dataset with point clouds and annotated 3D bounding boxes, you can create a new dataset class and train VoteNet on your own data. To ease the proces, some tips are provided in this [doc](https://github.com/facebookresearch/votenet/blob/master/doc/tips.md).


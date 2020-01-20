# NTU Computer Vision (Fall 2019) Final Project


## Getting Start
* Download pre-trained model first, you can find the download link from https://github.com/JiaRenChang/PSMNet and put the model in root folder of project


```
# 下載pretrain model[2] (KITTI 2012 pretrain model)
bash ./download.sh

or 

手動下載且放置在資料夾的根目錄 [備份連結]
https://drive.google.com/drive/folders/1Fkz9KJ6cNiXLxs9xdVQte4ACv8l-hC36?usp=sharing

# Inference
python3 main.py --input=DataFolderPath --output=OutputFolderPath

e.g.
python3 main.py --input=data --output=output
data為當前目錄下的資料夾, output會自動產生在當前目錄
```

### Cherry Pick

* Result for Synthetic 

| Ground Truth | Our Result |
|-------------|-------------|
|![](https://i.imgur.com/0DMubSu.png)|![](https://i.imgur.com/5Mh0wzE.png)|

* Result for Real(hava not ground truth) 

| Our Result |  
|-------------|  
|![](https://i.imgur.com/odIHaAy.png)|







### Citation
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```

###### Reference

* Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches (MC-CNN)
    * https://arxiv.org/abs/1510.05970
    * https://github.com/jzbontar/mc-cnn (lua)
    * https://github.com/Jackie-Chou/MC-CNN-python (tensorflow)
* End-to-End Learning of Geometry and Context for Deep Stereo Regression (GC-Net)
    * https://arxiv.org/abs/1703.04309
    * https://github.com/kelkelcheng/GC-Net-Tensorflow (tensorflow)
    * https://github.com/zyf12389/GC-Net (pytorch)
* Pyramid Stereo Matching Network (PSMNet - CVPR 2018)
    * https://arxiv.org/abs/1803.08669
    * https://github.com/JiaRenChang/PSMNet (Pytorch)
* GA-Net: Guided Aggregation Net for End-to-end Stereo Matching (GANet - CVPR2019)
    * https://arxiv.org/pdf/1904.06587.pdf
    * https://github.com/feihuzhang/GANet
* Real-Time Self-Adaptive Deep Stereo (CVPR2019 Oral)
    * https://arxiv.org/abs/1810.05424
    * https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo

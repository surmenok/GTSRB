# Overview #
A convolutional neural network for German traffic sign image classification.  

![](http://benchmark.ini.rub.de/Images/gtsrb/0.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/1.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/2.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/3.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/4.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/5.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/12.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/11.png)
![](http://benchmark.ini.rub.de/Images/gtsrb/8.png)

# Dataset #
[German Traffic Sign Recognition Dataset (GTSRB)](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=about) is an image classification dataset.  
The images are photos of traffic signs. The images are classified into 43 classes. The training set contains 39209 labeled images and the test set contains 12630 images. Labels for the test set are not published.  
See more details [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

# Model #
[ResNet-34](https://arxiv.org/abs/1512.03385) pretrained on ImageNet dataset, then finetuned on GTSRB dataset.

# Deep Learning Libraries #
[fastai](https://github.com/fastai/fastai/) with [PyTorch](http://pytorch.org/) backend.

# Metrics #
The model achieved 99.22% accuracy on the validation set (random 20% subset of the training dataset).

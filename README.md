# What's this
Implementation of Shake-Shake [[1]][Paper] by chainer


# Dependencies

    git clone https://github.com/nutszebra/shake_shake.git
    cd shake_shake
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 

# Details about my implementation

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy. 

* Learning rate schedule  
Initial learning rate is 0.1 and learning rate is divided by 10 at [60, 120. 160] epochs. The total number of epochs is 200.

* Optimization  
Momentum SGD with 0.9 momentum  

* Weight decay    
0.0005  

* Batch size  
128

* Cosine annealing  
Not implemented


# Cifar10 result

| network              | model  | total accuracy (%) |
|:---------------------|--------|-------------------:|
| [[1]][Paper]         | 2x32d  | 96.52              |
| [[1]][Paper]         | 2x64d  | 97.14              |
| [[1]][Paper]         | 2x96d  | 97.28              |
| my implementation    | 2x64d  | soon               |

<img src="https://github.com/nutszebra/shake_shake/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/shake_shake/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
Shake-Shake regularization of 3-branch residual networks [[1]][Paper]  
[paper]: https://openreview.net/forum?id=HkO-PCmYl "Paper"

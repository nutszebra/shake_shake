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

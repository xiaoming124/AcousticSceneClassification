import sys

sys.path.append('/usr/local/lib/python2.7/dist-packages')
from dnndk import n2cube, dputils
from ctypes import *
import numpy as np
import os
import threading
import time
import sys
from matplotlib import pyplot as plt
import matplotlib

features = np.zeros(shape=[40, 500])
img = features

wordlist = ['Indoor', 'Outdoor', 'Transportation']

### 配置DPU
KERNEL_CONV = "asc"
KERNEL_CONV_INPUT = "conv1"
KERNEL_FC_OUTPUT = "fc1000"

### 启动DPU
n2cube.dpuOpen()
kernel = n2cube.dpuLoadKernel(KERNEL_CONV)
task = n2cube.dpuCreateTask(kernel, 0)
channel = n2cube.dpuGetOutputTensorChannel(task, KERNEL_FC_OUTPUT)
FCResult = [0 for i in range(channel)]
mean = [104, 107, 123]

### 运行DPU并得到分类结果
dputils.dpuSetInputImage(task, KERNEL_CONV_INPUT, img, mean)
n2cube.dpuRunTask(task)
n2cube.dpuGetOutputTensorInHWCFP32(task, KERNEL_FC_OUTPUT, FCResult, channel)
label = FCResult.index(max(FCResult))

### 打印结果
print(label, wordlist[label])

### 关闭DPU
rtn = n2cube.dpuDestroyKernel(kernel)
n2cube.dpuClose()

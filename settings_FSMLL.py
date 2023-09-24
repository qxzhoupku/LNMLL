#====================#
#--> Description: 
#--> version: 
#--> Author: Wang Ziyu
#--> Date: 2023-03-18 18:33:02
#--> LastEditors: Wang Ziyu
#--> LastEditTime: 2023-03-18 18:54:37
#====================#
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

from numpy.fft import fft,ifft,fftshift,ifftshift
from matplotlib import cm
from scipy.special import jv

import os
import sys

# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['font.size'] = 14  #设置字体大小，全局有效

c0=299792458
pi=3.14159265359 # 圆周率
h=6.62607015e-34 # 普朗克常量
hbar=h/(2*pi) # 约化普朗克常量
k_B=1.380649e-23 # 玻尔兹曼常量

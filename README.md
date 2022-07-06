# 1. MNNDemos_Python

Python的示例

## 1.1 下载PyTorch GitHub模型

从PyTorch GitHub官网下载好mobilenet.py模型， pytorch 官网vision/torchvision/models/中的mobilenet.py文件。

https://github.com/pytorch/vision/tree/2ec0e847e813f6c8e060e67eb8886b4a0564c662/torchvision/models

## 1.2 测试mobilenet模型

`python mobilenet_test.py`

## 1.3 pytroch模型转onnx

`python model2onnx.py`


## 1.4 onnx模型转mnn,测试mnn模型

这是在conda的python环境中安装MNN。除了python能import调用API执行模型计算之外，还直接能在命令行操作MNN的工具。

```shell
pip install MNN
```

使用 mnnconvert -h 查看一些参数：

![](https://raw.githubusercontent.com/xddun/picgo_picture/main/20220630201746.png)

转换onnx模型：` mnnconvert -f ONNX --modelFile mobilenet_v2-b0353104.onnx --MNNModel mobilenet_v2-b0353104.mnn --bizCode MNN` ：

![](https://raw.githubusercontent.com/xddun/picgo_picture/main/20220630202220.png)

好玩的是执行`python mnn_test.py`遇到了错误，查阅资料发现是输入需要是NC4HW4的格式，改一下即可。

![](https://raw.githubusercontent.com/xddun/picgo_picture/main/20220630222104.png)



## 1.5 从源码编译安装MNN

pip中安装的onnx有时候转换模型并不是很好用，还是推荐从源码编译安装MNN。


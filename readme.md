# Background
银行卡号识别系统-基于Tensorflow&openCV

视频地址：[bilibili:【深度学习】银行卡号识别系统-基于TensorFlow&OpenCV](https://www.bilibili.com/video/BV1U7411i7rm)

输入用例：待识别银行卡图片

![image](https://github.com/zifeiyu0531/readme-imgs/blob/master/Bank-card-number-identification/%E8%BE%93%E5%85%A5%E6%A0%B7%E4%BE%8B.jpg)

输出用例：识别结果

![image](https://github.com/zifeiyu0531/readme-imgs/blob/master/Bank-card-number-identification/%E8%BE%93%E5%87%BA%E6%A0%B7%E4%BE%8B.png)

项目结构：
```
images: 训练集
test_images: 测试集
model: 训练模型
train.py: 入口文件
PreProcess.py & ImgHandle.py: 图像处理代码
forward.py: 深度学习模型前向传播代码
backward.py: 深度学习模型反向传播代码
app.py: 模型调用代码
```
# Enviroment
语言：`Python3.7`

深度学习框架：`TensorFlow`

图像处理：`OpenCV`
# Install
`Python3`运行环境：[下载地址](https://www.python.org/downloads/)

`TensorFlow`安装：
```
pip install tensorflow
```
`OpenCV`安装(清华镜像)：
```
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```
# Usage
Clone库到本地

### 训练模型
删除`model`文件夹内容

进入`train.py`

将布尔变量`train`的值改为`true`

运行`train.py`
### 模型调用
进入`train.py`

修改变量`file_path`的值为想要识别的图片路径

运行`train.py`
# Pack
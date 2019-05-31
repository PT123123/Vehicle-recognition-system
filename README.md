# Vehicle-recognition-system
需要安装的库：
PyQt5, cv2 ，hyperlpr
暂时不提供车型识别与颜色分类的模型
模型采用opencv DNN模块读取，所以确认你安装了含有DNN模块版本的cv2
![界面预览](https://github.com/PT123123/Vehicle-recognition-system/blob/master/png/1.jpg)

# 
车辆定位采用darknet yolov3在coco数据集上的预训练模型
车牌识别采用开源的hyperlpr：
https://github.com/zeusees/HyperLPR
视频播放界面基础：
https://github.com/fengtangzheng/pyqt5-opencv-video

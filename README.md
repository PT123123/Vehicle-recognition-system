
# get started：
1. PyQt5, 3.3以上的cv2 ，hyperlpr
2. 暂时不提供车型识别与颜色分类的模型
3. 下载 https://pjreddie.com/media/files/yolov3.weights ，并保存到yolo 目录下

# INTRO
![界面预览](https://github.com/PT123123/Vehicle-recognition-system/blob/master/png/demo.png)
模型采用opencv DNN模块读取，所以确认你安装了含有DNN模块版本(3.3以上)的cv2

1. 车辆定位采用darknet yolov3在coco数据集上的预训练模型
2. 车牌识别采用开源的hyperlpr：
https://github.com/zeusees/HyperLPR
3. 视频播放界面基础：
https://github.com/fengtangzheng/pyqt5-opencv-video

# TODO
连接KNN做颜色识别

#coding=utf-8
#created by pt,2019-3-18 16:31:56,
import os
import sys
import cv2
from hyperlpr import *#this is new plate recognition function package
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QStyle
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QObject,pyqtSignal,QThread,QMutex,QMutexLocker,Qt
from UI_file import *

import numpy as np
import time
import threading
from queue import Queue

#get current path of this file
CURPATH=os.path.dirname(os.path.realpath(__file__))

#many queues use to communicate around different threads
color_q=Queue()
plate_q=Queue()
type_q=Queue()
pinpai_q=Queue()
pinpai_img_q=Queue()
im_q=Queue()
img_car_q=Queue()
to_color_q=Queue()
time_q=Queue()

#used to control different thread by calling event.set and event.wait
even_yolo=threading.Event()
even_model=threading.Event()
even_color=threading.Event()
even_license=threading.Event()

#thread of model classification,not implemented
class MODEL_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
	#model_thread_running used to stop the thread when needed
        self.model_thread_running=True
        #self.daemon = True
        '''
        self.classes=('background','richan','jipu','benchi','bentian','fengtian','aodi','wuling','baoma','dazhong',
'qirui','xuefulan','lufeng','futian','mazida','dongfeng','sibalu','biaozhi','jianghuai',
'xiandai','ouge','tianye','changcheng','changan','zhongtai','qiya','sanling','bieke',
'xuetielong','changhe','sikeda','haima','biyadi','shuanglong','lingmu','guangqijiao',
'woerwo','yiweike','zhonghua','feiyate','changanshangyong','mingjue','dongnan','jinbei',
'baoshijie','lifan','leikesasi','yiqi','luhu','fute','rongwei','leinuo','fudi','kairui',
'qichen','datong','jiebao','hafu','lianhua','kaidilake','jiangling','huanghai','kelaisilei',
'baojun','yingfeinidi','hafei','mini')
        self.modelPATH=[os.path.join(CURPATH,'lib/model/deploy.prototxt'),
		os.path.join(CURPATH,'lib/model/VGG_VOC2007_SSD_300x300_iter_20710.caffemodel')]
        self.model=cv2.dnn.readNetFromCaffe(*self.modelPATH)
        '''
    def run(self):
        global what_pt_want  
        print("pinpai functionality loaded successfully-------------------------------------")
        while self.model_thread_running:
            even_model.wait()
            confid_result=[]
            class_result=[]
            if not pinpai_img_q.empty():
                img=pinpai_img_q.get()
            else:
                nd.settable('ERROR',False,False,False,False)
                continue
            nd.settable('NOT IMPLEMENTED',False,False,False,False)#added
            '''
            height=img.shape[0]
            width=img.shape[1]
            blob =cv2.dnn.blobFromImage(img,1.0,(300,300),(104,117,123),False,False)
            self.model.setInput(blob)
            detections=self.model.forward()

            for i in np.arange(0, detections.shape[2]):
                # extract the confidence
                confidence = detections[0, 0, i, 2]
                if confidence > 0.1:#confidence
                    # extract index of class label
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    class_result.append(self.classes[idx])
                    confid_result.append(confidence* 100)
            if not len(confid_result) == 0:
                location=confid_result.index(max(confid_result))
                nd.settable(class_result[location],False,False,False,False)
            else:
                nd.settable('NONE',False,False,False,False)
            even_model.clear()
            '''
    def close(self):
        pass

#thread of vehicle detection, trained with darknet yolo, this repo provide coco pretrained model

class YOLO_thread(threading.Thread):
    #init by loading the model file and load it using opencv DNN 
    #get detecting bounding box area by loading config.ini, this para is in line 2,which mean index by [1]
    def __init__(self):
        threading.Thread.__init__(self)
        self.yolo_thread_running=True
        #self.daemon = True
        weightsPath=os.path.join(CURPATH,'yolo/yolov3.weights')
        configPath=os.path.join(CURPATH,'yolo/yolov3.cfg')
        labelsPath = os.path.join(CURPATH,'yolo/coco.names')
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.BOUNDING = open(os.path.join(CURPATH,'config/config.ini'),'r').read().strip().split("\n")[1].split('-')
        for _ in range(len(self.BOUNDING)):
            self.BOUNDING[_]=float(self.BOUNDING[_])
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
   #detect img sent by im_q, sift out unneccessary results by threshhold and NMS(Non-Max-suppresion)
   #if results is within the insterested area described by config self.BOUNDING ,crop target area and awaken other thread of classification
    def run(self):
        global what_pt_want
        global yoloimg
        print('yolo warmed up')
        while self.yolo_thread_running:
            even_yolo.wait()
            _=time.time()
          #emptify if queue is not empty
            if not pinpai_q.empty():
                pinpai_q.get()
            if not color_q.empty():
                color_q.get()
            if not plate_q.empty():
                plate_q.get()
            if not type_q.empty():
                type_q.get()
            pt_dict={}
            boxes = []
            confidences = []
            classIDs = []
            image = cv2.cvtColor(im_q.get(), cv2.COLOR_RGB2BGR)
            (H, W) = image.shape[:2]
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            self.net.setInput(blob)
            layerOutputs = self.net.forward(ln)
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    if self.LABELS[classID] not in ('truck','car','bus','train'):#'motorbike'
                        continue
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        #left upper
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # nms
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    #, confidences[i]
                    type_q.put(self.LABELS[classIDs[i]])
                    expression1=image.shape[0]*self.BOUNDING[0]<(int(x)+int(x+w))/2<image.shape[1]*self.BOUNDING[1]
                    expression2=image.shape[2]*self.BOUNDING[3]<(int(y)+int(y+h))/2<image.shape[0]*self.BOUNDING[3]
                    if expression1:
                        if expression2:
                            if h<0 or w<0 or y<0 or x<0:
                                continue
                            nd.settable(False,False,self.LABELS[classIDs[i]],False,False)
                            img_crop=image[y:y+h,x:x+w]
                            save_time=time.asctime(time.localtime(time.time())).replace(' ','_').replace(':','_')
                            flag_=False
                            flag_=cv2.imwrite(os.path.join(CURPATH,'archives/{}.jpg'.format(save_time)),img_crop)#save vehicle image by name of time
                            if not flag_:
                                print('img saving error')
                            nd.settable(False,False,False,False,save_time)                       
                            #########################
                            pinpai_img_q.put(img_crop)
                            to_color_q.put(img_crop)
                            img_car_q.put(img_crop)
                            even_color.set()
                            even_model.set()
                            even_license.set() 
                            break#tem
			    #TODO:work on this 'break'
    def close(self):
        self.yolo_thread_running=False
        print('closing yolo session')


#license plate recognition,u should install it by pip install hyperlpr first
class LICENSE_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.license_thread_running=True
        #self.daemon = True
    def run(self):
        PATH=os.path.dirname(hyperlpr.__file__)#CURPATH,'lib/hyperlpr/models')
        PATH=os.path.join(PATH,'models')
        PR = hyperlpr.LPR(PATH)
        def HyperLPR_PlateRecogntion(Input_BGR,minSize=30,charSelectionDeskew=True):
            return PR.plateRecognition(Input_BGR,minSize,charSelectionDeskew)
        #so called warm up
        license_warmup=np.zeros([400,400,3],np.uint8)
        HyperLPR_PlateRecogntion(license_warmup)
        del license_warmup
        while self.license_thread_running:
            even_license.wait()
            image=img_car_q.get()
            tem=HyperLPR_PlateRecogntion(image)
            print(type(tem))
            plate="".join('%s' %id for id in tem)
            if plate!='':
                nd.settable(False,plate.split(',')[0].replace('[',''),False,False,False)
            else:
                nd.settable(False,'NONE',False,False,False)
    def close(self):
        self.license_thread_running=False
        print('closing license')

#color classification, TODO:use KNN to implement this 
class COLOR_thread(threading.Thread):
    #thread-2
    def __init__(self):
        threading.Thread.__init__(self)
        self.color_thread_running=True
        '''
        #self.daemon = True
        #modelRecognitionPath = [os.path.join(CURPATH,'lib/color_model/moxing/deploy.prototxt'),
        #            os.path.join(CURPATH,'lib/color_model/moxing/color.caffemodel')]            #color_train_iter_20000.caffemodel"]
        #self.modelRecognition = cv2.dnn.readNetFromCaffe(*modelRecognitionPath)
        #self.color = ('brown','grey','white','pink','purple','red','green','blue','yellow','black')
        '''
    def run(self):
        while self.color_thread_running:
            even_color.wait()
            img_crop=to_color_q.get()
            nd.settable(False,False,False,'NOT IMPLEMENTED',False)
            try:
                blob = cv2.dnn.blobFromImage(cv2.cvtColor(np.asarray(img_crop),cv2.COLOR_RGB2HSV), 1.0, (227, 227),(81.5,48.5,102.4), False, False)
                self.modelRecognition.setInput(blob)
                a = np.asarray(self.modelRecognition.forward()[0])
                nd.settable(False,False,False,self.color[np.argmax(a)],False)
            except:
                nd.settable(False,False,False,'failed',False)
    def close(self):
        self.color_thread_running=False
        print('color closed')
            
class mywindow(QtWidgets.QWidget,Ui_Dialog):
    global what_pt_want
    status = 0
    video_type = 0
    TYPE_VIDEO = 0
    TYPE_CAMERA = 1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2
    global pt_video_counter

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        ############################
        self.yolo_thread=YOLO_thread()
        self.license_thread=LICENSE_thread()
        self.model_thread=MODEL_thread()
        self.color_thread=COLOR_thread()
        self.color_thread.start()
        self.yolo_thread.start()
        self.license_thread.start()
        self.model_thread.start()
        ###########################
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.pinpai_count=0
        self.color_count=0
        self.type_count=0
        self.plate_count=0
        self.time_count=0
        self.line_counter=0
        self.table_dict={}
        self.table_dict['time']=Queue()
        self.table_dict['pinpai']=Queue()
        self.table_dict['color']=Queue()
        self.table_dict['plate']=Queue()
        self.table_dict['type']=Queue()
        

        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)#once timer emit a signal,run show_video_images()
        self.INTERVAL = int(open(os.path.join(CURPATH,'config/config.ini')).read().strip().split("\n")[0])
        self.VIS_RECGTANGLE = int(open(os.path.join(CURPATH,'config/config.ini')).read().strip().split("\n")[2])
    def openimage(self):
        global pt_video_counter
        self.video_type=self.TYPE_VIDEO
        self.reset()
        if 'self.playCapture' in locals() or 'self.playCapture' in globals():
            self.playCapture.release()
        imgName,imgType= QFileDialog.getOpenFileName(self,"open the image",""," All Files (*);;*.asf;;*.mp4;;*.mpg;;*.avi")
        global what_pt_want
        what_pt_want=imgName
        if 'what_pt_want' in locals() or 'what_pt_want' in globals():
            pass
            if what_pt_want == "" or what_pt_want is None:
                return
            else:
                self.pushButton.setEnabled(True)
                self.playCapture = cv2.VideoCapture()
                pt_video_counter=1
                self.playCapture.open(what_pt_want)
                fps = self.playCapture.get(cv2.CAP_PROP_FPS)#used to be cv2.CAP_PROP_FPS
                self.timer.set_fps(fps)
                self.timer.start()
                #self.timer.stopped=False 
                self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.status = (self.STATUS_PLAYING,self.STATUS_PAUSE,self.STATUS_PLAYING)[self.status] 
        else:
            return

    def settable_function(self,type_signal,result):
        #self.srollBar=self.tableWidget
        self.table_dict[type_signal].put(result)
        if not (self.table_dict['time'].empty() or self.table_dict['type'].empty() or self.table_dict['color'].empty() 
		or self.table_dict['plate'].empty() or self.table_dict['pinpai'].empty()):
            pinpai=self.table_dict['pinpai'].get()
            plate=self.table_dict['plate'].get()
            time= self.table_dict['time'].get()
            type= self.table_dict['type'].get()
            color=self.table_dict['color'].get()
            if not pinpai ==plate:
                self.tableWidget.insertRow(self.line_counter)
                self.tableWidget.setItem(self.line_counter,0,QTableWidgetItem(time))
                self.tableWidget.setItem(self.line_counter,1,QTableWidgetItem(pinpai))
                self.tableWidget.setItem(self.line_counter,2,QTableWidgetItem(color))
                self.tableWidget.setItem(self.line_counter,3,QTableWidgetItem(type))
                self.tableWidget.setItem(self.line_counter,4,QTableWidgetItem(plate))
                self.line_counter+=1
       
   
    def reset(self):
        self.timer.stopped=True
        if 'self.playCapture' in locals() or 'self.playCapture' in globals():
            self.playCapture.release()
        self.status = self.STATUS_INIT
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
	
    #displaying video frame
    def show_video_images(self):
        '''it detected a car of interst in wanted region,after inference,run this function'''
        #if video is open successfully,read it
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            #,frame is a ndarray,frame.shape index 0 and 1 stand for height and width
            if success:
                if self.VIS_RECGTANGLE:
                    cv2.rectangle(frame,(int(frame.shape[1]*self.yolo_thread.BOUNDING[0]),int(frame.shape[0]*self.yolo_thread.BOUNDING[2])),
					(int(frame.shape[1]*self.yolo_thread.BOUNDING[1]),int(frame.shape[0]*self.yolo_thread.BOUNDING[3])),(0,255,0),2)
                global pt_video_counter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame.shape[:2]
                temp_image = QImage(frame.flatten(), width, height,QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                temp_pixmap=temp_pixmap.scaled(self.graphicsView.width(),self.graphicsView.height())
                self.graphicsView.setPixmap(temp_pixmap)
                pt_video_counter+=1
                if (pt_video_counter%(int)(self.timer.frequent*self.INTERVAL)==0):#INTERVAL
                    if True:#not even_yolo.is_set():
                        if True:#not even_model.is_set():
                            im_q.put(frame)
                            even_yolo.set()#test3.6                                                                           
            else:
                print("read failed, no frame data")
                self.reset()
               
        else:
            print('end')
            self.reset()#open file or capturing device error, init again

    def webcamera(self):
        self.reset()
        global pt_video_counter
        if not self.pushButton_3.isEnabled():
            return
        else:
            self.pushButton.setEnabled(True)
        self.playCapture = cv2.VideoCapture(0)
        fps = self.playCapture.get(cv2.CAP_PROP_FPS)#used to be cv2.CAP_PROP_FPS
        if fps ==0:
            QMessageBox.warning(self,'error','no webcamera')
            return
        self.timer.set_fps(fps) 
        self.video_type = self.TYPE_CAMERA
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        pt_video_counter=1
        self.timer.start()
        self.status = (self.STATUS_PLAYING,self.STATUS_PAUSE,self.STATUS_PLAYING)[self.status]
	
    #run a video 
    def inquiry(self):
        global pt_video_counter
        if not self.pushButton.isEnabled():
            return
        if self.status is self.STATUS_INIT:
            pass
        elif self.status is self.STATUS_PLAYING:
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stopped=True 
            if self.video_type is self.TYPE_CAMERA:
                self.playCapture.release()
        elif self.status is self.STATUS_PAUSE:
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            if self.video_type is self.TYPE_CAMERA:  
                self.webcamera()
            else:
                self.timer.start()
        self.status = (self.STATUS_PLAYING,self.STATUS_PAUSE,self.STATUS_PLAYING)[self.status]   

    def closeEvent(self,event):
        reply = QMessageBox.question(self,'warning',"u sure u wanna quit?",QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
        if reply == QMessageBox.Yes:
            guanbudiaoa
            event.accept()
            self.reset()
            self.yolo_thread.close()
            self.model_thread.close()
            self.color_thread.close()
            self.license_thread.close()
            self.timer.wait()
            nd.wait()
            pass
            sys.exit(app.exec_())
        else:
            event.ignore()


#run this method when dobble click one table line
    def display_table(self):
        line=self.tableWidget.currentRow()
        value=self.tableWidget.item(line, 0).text()
        image_file=os.path.join(CURPATH,'archives/{}.jpg'.format(value))
        if not os.path.exists(image_file):
            QMessageBox.about(self,'error','img not exists')
        result_image=QtGui.QPixmap(image_file).scaled(window.graphicsView_frame.width(), window.graphicsView_frame.height())
        window.graphicsView_frame.setPixmap(result_image)
    
class Communicate(QObject):
    signal = pyqtSignal(str)   
    
#https://blog.csdn.net/qq_34710142/article/details/80936986 
class VideoTimer(QThread):
    def __init__(self, frequent=100):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()#the lock between threads

    def run(self):#method run just play the pix of the video one by one
        with QMutexLocker(self.mutex):
            self.stopped = False
        self.run_()
    def run_(self):
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)
    #def stop(self):
    #    with QMutexLocker(self.mutex):
    #        self.stopped = True
    def set_fps(self, fps):
        self.frequent = fps   
	
#Qthread used to display table widget
class Network_daemon(QThread):
    '''daemon thread, function haha is used to display brand, license plate number,color and model'''
    trigger_table = pyqtSignal(str,str) 
    def __int__(self):
        super(Network_daemon, self).__init__()
    def run(self): 
        return  
    def settable(self,pinpaistr,platestr,typestr,colorstr,timestr): 
        #run the inquiry1
        if pinpaistr is not False:
            pinpai_q.put(pinpaistr)
            self.trigger_table.emit('pinpai',pinpai_q.get())
        if platestr is not False:
            plate_q.put(platestr)
            self.trigger_table.emit('plate',plate_q.get())
        if typestr is not False:
            type_q.put(typestr)
            self.trigger_table.emit('type',type_q.get())
        if colorstr is not False:
            color_q.put(colorstr)
            self.trigger_table.emit('color',color_q.get())
        if timestr is not False:
            time_q.put(timestr)
            self.trigger_table.emit('time',time_q.get())
 
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    nd=Network_daemon()
    nd.trigger_table.connect(window.settable_function)
    nd.start()
    window.show()
    window.pushButton_2.setEnabled(True)
    sys.exit(app.exec_())


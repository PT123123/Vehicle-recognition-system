#coding=utf-8
#created by pt,2019-3-18 16:31:56,
import os
import sys
import cv2
from lib.hyperlpr import hyperlpr#this is new plate recognition function package
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QStyle
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QObject,pyqtSignal,QThread,QMutex,QMutexLocker,Qt
from UI_file import *
import numpy as np
import time
import threading
from queue import Queue
import xlwt
import ctypes
CURPATH=os.path.dirname(os.path.realpath(__file__))
try:
    temp = ctypes.windll.LoadLibrary(os.path.join(CURPATH,'opencv_ffmpeg410_64.dll'))
except:
    print('opencv')
color_q=Queue()#used to pass the result of color
plate_q=Queue()#used to pass the result of plate recognition
type_q=Queue()#same as above
pinpai_q=Queue()
pinpai_img_q=Queue()# used to pass the cv2-format img between threads
im_q=Queue()
img_car_q=Queue()
to_color_q=Queue()
time_q=Queue()

even_yolo=threading.Event()# to synchronize between tasks
even_model=threading.Event()
even_color=threading.Event()
even_license=threading.Event()
##########################################################################################

def compute_IOU(rec1,rec2):
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
        
        
def transfer_time_format(str):
    if "_" in str:
        _ = str.split("_")
        return _[0]+"时"+_[1]+"分"+_[2]+"秒"+_[3]+"年"+_[4]+"月"+_[5]+"日"
    else:
        #print(str.replace("年","_").replace("月","_").replace("日","").replace("时","_").replace("分","_").replace("秒","_"))
        return str.replace("年","_").replace("月","_").replace("日","").replace("时","_").replace("分","_").replace("秒","_")
#https://www.jianshu.com/p/7b6a80faf33f
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
        '''#暂不提供车牌功能,注释掉
    def run(self):
        global what_pt_want  
        print("pinpai functionality loaded successfully-------------------------------------")
        while self.model_thread_running:
            even_model.wait()
            confid_result=[]
            class_result=[]
            time.sleep(1.0)#buffer
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

class YOLO_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.yolo_thread_running=True
        #self.daemon = True
        weightsPath=os.path.join(CURPATH,'lib/yolo/yolov3.weights')
        configPath=os.path.join(CURPATH,'lib/yolo/yolov3.cfg')
        labelsPath = os.path.join(CURPATH,'lib/yolo/coco.names')
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.BOUNDING = [0.300,0.600,0.300,0.600]
        self.IOU=0.1
    def run(self):
        global what_pt_want
        global yoloimg
        print('yolo done')
        while self.yolo_thread_running:
            even_yolo.wait()
          #emptify if queue is not empty
            if not pinpai_q.empty():
                pinpai_q.get()
            if not color_q.empty():
                color_q.get()
            if not plate_q.empty():
                plate_q.get()
            if not type_q.empty():
                type_q.get()
###############################################################
            confid_result=[]
            class_result=[]
            classIDs=[]
            if im_q.empty():
                time.sleep(0.5)
                continue
            img = cv2.cvtColor(im_q.get(), cv2.COLOR_RGB2BGR)
            (H, W) = img.shape[:2]
            ln = self.net.getLayerNames()
            ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            self.net.setInput(blob)
            layerOutputs = self.net.forward(ln)
            boxes=[]
            confidences=[]
            for output in layerOutputs:
                # 对每个检测进行循环
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    if self.LABELS[classID] not in ('truck','car','bus','train'):#'motorbike'
                        continue
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        #边框的左上角
                        if centerX<0 or centerY<0 or width<0 or height<0:
                            continue
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # 极大值抑制
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    #, confidences[i]
                    type_q.put(self.LABELS[classIDs[i]])
                    rec_=[W*self.BOUNDING[0],H*self.BOUNDING[2],W*self.BOUNDING[1],H*self.BOUNDING[3]]
                    rec_car=[x,y,x+w,y+h]
                    IOU = compute_IOU(rec_,rec_car)
                    print(IOU)
                    expression1=True
                    expression2=IOU>self.IOU
                    if expression1:
                        if expression2:
                            nd.settable(False,False,self.LABELS[classID],False,False)
                            img_crop=img[y:y+h,x:x+w]
                            ################################
                            save_time = time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日', h='时', f='分', s='秒')
                            flag_=cv2.imwrite(os.path.join(CURPATH,'archives/{}.jpg'.format(transfer_time_format(save_time))),img_crop)#save vehicle image by name of time
                            if not flag_:
                                print('图片保存失败')
                            nd.settable(False,False,False,False,save_time)                       
                            #########################
                            pinpai_img_q.put(img_crop)
                            to_color_q.put(img_crop)#img_crop is a PIL format image
                            img_car_q.put(img_crop)
                            if not even_model.is_set():
                                even_model.set()
                            even_color.set()
                            even_license.set()
                            even_yolo.clear()
                            break#tem
    def close(self):
        self.yolo_thread_running=False
        print('closing yolo session')



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
            plate="".join('%s' %id for id in tem)
            if plate!='':
                nd.settable(False,plate.split(',')[0].replace('[',''),False,False,False)
            else:
                nd.settable(False,'NONE',False,False,False)
    def close(self):
        self.license_thread_running=False
        print('closing license')


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
        '''#颜色模型已去除,暂不提供,请fork后自行DIY
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
            
class mywindow(Ui_Dialog):
    global what_pt_want
    status = 0
    video_type = 0
    TYPE_VIDEO = 0
    TYPE_CAMERA = 1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2
    CAMID='qq'
    BEFORE='0'
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
        self.line_counter=0
        self.table_dict={}
        self.table_dict['time']=Queue()
        self.table_dict['pinpai']=Queue()
        self.table_dict['color']=Queue()
        self.table_dict['plate']=Queue()
        self.table_dict['type']=Queue()
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_video_images)#once timer emit a signal,run show_video_images()
        self.INTERVAL = 1
        self.VIS_RECGTANGLE = 0
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
                if fps ==0:
                    QMessageBox.warning(self,'error','fps不能为0')
                    return
                else:
                        self.timer.set_fps(fps)
                self.timer.start()
                #self.timer.stopped=False 
                self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.status = (self.STATUS_PLAYING,self.STATUS_PAUSE,self.STATUS_PLAYING)[self.status] 
        else:
            return

    def settable_function(self,type_signal,result):
        self.table_dict[type_signal].put(result)
        #print(type_signal,' :   ',self.table_dict[type_signal].qsize())
        if not (self.table_dict['time'].empty() or self.table_dict['type'].empty() or self.table_dict['color'].empty() 
		or self.table_dict['plate'].empty() or self.table_dict['pinpai'].empty()):
            pinpai=self.table_dict['pinpai'].get()
            
            plate=self.table_dict['plate'].get()
            time= self.table_dict['time'].get()
            type= self.table_dict['type'].get()
            color=self.table_dict['color'].get()
            if self.BEFORE == plate.strip():
            	return#6_13.P.M.
            else:
                pass
                #print(plate.strip()+'    '+self.BEFORE)
            if not pinpai ==plate:
                self.tableWidget.insertRow(self.line_counter)
                self.tableWidget.setItem(self.line_counter,0,QTableWidgetItem(time))
                self.tableWidget.setItem(self.line_counter,1,QTableWidgetItem(pinpai))
                self.tableWidget.setItem(self.line_counter,2,QTableWidgetItem(color))
                self.tableWidget.setItem(self.line_counter,3,QTableWidgetItem(type))
                self.tableWidget.setItem(self.line_counter,4,QTableWidgetItem(plate))
                self.line_counter+=1
                self.tableWidget.verticalScrollBar().setValue(self.line_counter)
                self.BEFORE=plate.strip()
       
   
    def reset(self):
        self.timer.stopped=True
        if 'self.playCapture' in locals() or 'self.playCapture' in globals():
            self.playCapture.release()
        self.status = self.STATUS_INIT
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def show_video_images(self):
        '''it detected a car of interst in wanted region,after inference,run this function'''
        #if video is open successfully,read it
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            #,frame is a ndarray,frame.shape index 0 and 1 stand for height and width
            if success:
                if self.VIS_RECGTANGLE:
                    cv2.rectangle(frame,(int(frame.shape[1]*self.yolo_thread.BOUNDING[0]),int(frame.shape[0]*self.yolo_thread.BOUNDING[2])),
					(int(frame.shape[1]*self.yolo_thread.BOUNDING[1]),int(frame.shape[0]*self.yolo_thread.BOUNDING[3])),(0,255,0),3)
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
                            im_q.put(frame)#6_16,modified
                            even_yolo.set()                                                                          
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
        self.playCapture = cv2.VideoCapture(self.CAMID.strip())
        fps = self.playCapture.get(cv2.CAP_PROP_FPS)#used to be cv2.CAP_PROP_FPS
        if fps ==0:
            QMessageBox.warning(self,'error','fps不能为0')
            return
        self.timer.set_fps(fps) 
        self.video_type = self.TYPE_CAMERA
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        pt_video_counter=1
        self.timer.start()
        self.status = (self.STATUS_PLAYING,self.STATUS_PAUSE,self.STATUS_PLAYING)[self.status]

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
            for filepath in os.listdir(os.path.join(CURPATH,'archives')):
                os.remove(os.path.join(os.path.join(CURPATH,'archives'),filepath))
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

    def display_table(self):
        line=self.tableWidget.currentRow()
        value=self.tableWidget.item(line, 0).text()
        image_file=os.path.join(CURPATH,'archives/{}.jpg'.format(transfer_time_format(value)))
        if not os.path.exists(image_file):
            QMessageBox.about(self,'error','图片不存在')
            return
        result_image=QtGui.QPixmap(image_file).scaled(window.graphicsView_frame.width(), window.graphicsView_frame.height())
        window.graphicsView_frame.setPixmap(result_image)
    def export_txt(self):
        save_path=QFileDialog.getSaveFileName(self,'save file',CURPATH,'txt(*txt)')
        save_path=save_path[0]
        if not save_path.endswith('.xls'):
            save_path = save_path+'.txt'
        try:
            predix=' 保存时间：'+time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日', h='时', f='分', s='秒')
            if not os.path.exists(os.path.join(CURPATH,save_path)):
                predix='%s'%('时间')+ '%45s'%('车标')+'%20s'%('颜色')+'%15s'%('车型')+'%25s'%('车牌')+'\n'+predix
            with open(os.path.join(CURPATH,save_path),'a+') as f:
                f.write(predix+'\n')
                for line in range(self.line_counter):
                    values=[]
                    values.append('%20s'%(self.tableWidget.item(line, 0).text()))
                    values.append('%10s'%(self.tableWidget.item(line, 1).text()))
                    values.append('%10s'%(self.tableWidget.item(line, 2).text()))
                    values.append('%10s'%(self.tableWidget.item(line, 3).text()))
                    values.append('%5s'%(self.tableWidget.item(line, 4).text()))
                    f.write('      '.join(values)+'\n')
            QMessageBox.information(self,'Great！','已成功保存为%s'%(save_path))
        except Exception as e:
            print(repr(e))
            fname='error.txt'
            predix=' 出错时间：'+time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日', h='时', f='分', s='秒')
            with open(fname, 'a+') as f:
                f.write('\n'+repr(e))
            QMessageBox.warning(self,'save error!!','  save error message to {}'.format(fname))
    def export(self):#先用.xls格式保存结果
        save_path=QFileDialog.getSaveFileName(self,'save file',CURPATH,'xls(*xls)')
        save_path=save_path[0]
        if not save_path:
            return
        try:
            workbook = xlwt.Workbook(encoding = 'utf-8')
            worksheet = workbook.add_sheet('My worksheet',cell_overwrite_ok=True)
            if not os.path.exists(os.path.join(CURPATH,save_path)):
                pass
            _=0
            for content_ in ['时间','车标','颜色','车型','车牌']:
                worksheet.write(0,_,label=content_)
                _+=1
            for line in range(self.line_counter):
                for _ in range(5):
                    worksheet.write(line+1, _ ,label=self.tableWidget.item(line,_).text())
                if not save_path.endswith('.xls'):
                    save_path = save_path+'.xls'
                workbook.save(save_path)
        except Exception as e:
            print(repr(e))
            QMessageBox.warning(self,'保存失败',repr(e))
            return
        QMessageBox.information(self,'保存成功',' 已保存到 %s'%(str(save_path)))
    
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
    def set_fps(self, fps):
        self.frequent = fps

class Network_daemon(QThread):
    '''daemon thread, function haha is used to display brand, license plate number,color and model'''
    trigger_table = pyqtSignal(str,str) 
    def __int__(self):
        super(Network_daemon, self).__init__()
    def run(self):
        while True:
            time.sleep(5)
            if not pinpai_img_q.empty():
                if not even_model.is_set():
                    even_model.set()
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
    global window
    window = mywindow()
    nd=Network_daemon()
    nd.trigger_table.connect(window.settable_function)
    nd.start()
    window.show()
    window.pushButton_2.setEnabled(True)
    sys.exit(app.exec_())


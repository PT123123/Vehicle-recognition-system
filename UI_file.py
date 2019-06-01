#_*_ coding:gbk _*_

# Form implementation generated from reading ui file 
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

class Ui_Dialog(object):
    def setupUi(self, Dialog):#Dialog is an instance objec of class QtWidgets.QWidget
        self.desktop = QApplication.desktop()
        #get monitor resolution
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()

        global_widget=QWidget(self)

        global_layout = QHBoxLayout(global_widget)

        Dialog.setObjectName("Dialog")
        Dialog.resize(1518, 844)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)        
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)

        self.pushButton.setObjectName("pushButton")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4.setObjectName("pushButton_4")
###########################
        #self.listWidget.verticalScrollBar().valueChanged.connect(lambda :print(1))
        #todo:above is roll down automatically
        self.spacerItem = QSpacerItem(2, 20,QSizePolicy.Expanding,QSizePolicy.Expanding)
        

        self.button_space= QtWidgets.QHBoxLayout()
        self.button_space2= QtWidgets.QHBoxLayout()

        self.button_space.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton)
        self.button_space.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton_2)
        self.button_space.addItem(self.spacerItem)
        
        self.button_space2.addItem(self.spacerItem)
        self.button_space2.addWidget(self.pushButton_3)
        self.button_space2.addItem(self.spacerItem)
        self.button_space2.addWidget(self.pushButton_4)
        self.button_space2.addItem(self.spacerItem)
   
   
        self.button_layout=QVBoxLayout()
        for i in (self.button_space,self.button_space2):
            self.button_layout.addLayout(i)
            
            
        #########################################
        self.graphicsView = QtWidgets.QLabel(Dialog)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setTextFormat(QtCore.Qt.RichText)#added
        self.graphicsView.setFixedWidth(self.width*2/5)
        self.graphicsView.setFixedHeight(self.height/2)
        self.graphicsView.setFrameShape(QtWidgets.QFrame.Box)

        #add buttons_widget and Qgraphics to vertical layout
        veritical_layout_graphics_buttons = QtWidgets.QVBoxLayout()
        veritical_layout_graphics_buttons.addWidget(self.graphicsView)
        veritical_layout_graphics_buttons.addLayout(self.button_layout)

        self.graphicsView_frame = QtWidgets.QLabel(Dialog)
        self.graphicsView_frame.setObjectName("graphicsView_frame")
        self.graphicsView_frame.setTextFormat(QtCore.Qt.RichText)#added
        self.graphicsView_frame.setFixedHeight(self.height/3)
        self.graphicsView_frame.setFixedWidth(self.width/3)
        self.graphicsView_frame.setFrameShape(QtWidgets.QFrame.Box)
        #table configuration:
        #https://blog.csdn.net/zhulove86/article/details/52599738/
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)#can only choose one line
        self.tableWidget.setEditTriggers(QTableView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.doubleClicked.connect(self.display_table)
        #self.tableWidget.item.setTextAlignment(Qt.AlignHCenter)
        self.tableWidget.setHorizontalHeaderLabels(['time','make','color','type','license',' '])
        ##################
        VerticalLayout = QtWidgets.QVBoxLayout()
        VerticalLayout.addWidget(self.tableWidget)
        VerticalLayout.addWidget(self.graphicsView_frame)
 
        #######################################
        #############################

        global_layout.addLayout(veritical_layout_graphics_buttons)       
        global_layout.addLayout(VerticalLayout)
        
        self.retranslateUi(Dialog)
        self.pushButton_2.clicked.connect(Dialog.openimage)
        self.pushButton.clicked.connect(Dialog.inquiry)#inquiry
        self.pushButton_3.clicked.connect(Dialog.webcamera)#inquiry
        self.pushButton_4.clicked.connect(Dialog.export)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.setLayout(global_layout)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Vehicle Recognition System"))
        self.pushButton.setText(_translate("Dialog", "   "))
        self.pushButton_2.setText(_translate("Dialog", "Load Video File"))
        self.pushButton_3.setText(_translate("Dialog", "Start Webcam"))
        self.pushButton_4.setText(_translate("Dialog", "Export"))













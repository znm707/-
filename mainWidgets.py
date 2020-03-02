"""
    GUI窗口模块
"""
# coding=utf-8
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
import os
import cv2
import numpy as np
import Process
import time


class mainwidget(QWidget):
    """
        主窗口类，包含UI与交互逻辑
    """
    def __init__(self):
        super().__init__()
        self.setGeometry(50, 20, 1320, 600)
        self.setWindowTitle('曲线识别')
        self.move(10, 20)
        self.initUI()

    def initUI(self):
        """
            初始化UI
        """
        self.cwd = os.getcwd()
        # 输出文本框
        self.textEdit = QTextEdit(self)
        self.textEdit.setGeometry(0, 0, 450, 300)
        self.textEdit.move(840, 60)

        # 清空按钮
        self.clearbtn = QPushButton('清空',self)
        self.clearbtn.move(840, 20)
        self.clearbtn.clicked.connect(self.cleartext)

        # 加载按钮
        self.loadbtn = QPushButton('加载图片', self)
        self.loadbtn.move(20, 20)
        self.loadbtn.clicked.connect(self.loadimg)

        # 处理按钮
        self.processbtn = QPushButton('处理', self)
        self.processbtn.move(330, 20)
        self.processbtn.released.connect(self.ImgProcess)

        # 加载图片Label
        self.loadlabel = QLabel(self)
        self.loadlabel.setFixedSize(384, 512)
        self.loadlabel.move(20, 60)
        self.loadlabel.setStyleSheet("border: 1px solid white")

        # 显示处理图片Label
        self.showlabel = QLabel(self)
        self.showlabel.setFixedSize(384, 512)
        self.showlabel.move(420, 60)
        self.showlabel.setStyleSheet("border: 1px solid white")

    def reshapeImg(self, img):
        """
            将图片修改为512X384
        """
        height = img.shape[0]
        width = img.shape[1]
        if width > height:
            img = np.rot90(img, 1)
        img = cv2.resize(img, (384, 512), interpolation = cv2.INTER_AREA)
        return img

    def RgbCv2QPixmap(self, img):
        """
            RGB图片转化为QPixmap
        """
        img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = img_rbg.shape
        bytesPerLine = 3 * width
        QImg = QImage(img_rbg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)
        return pixmap

    def GrayCv2QPixmap(self, gray):
        """
            灰度图片转化为QPixmap
        """
        height, width = gray.shape
        bytesPerLine = width
        QImg = QImage(gray.data, width, height, bytesPerLine, QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(QImg)
        return pixmap

    def Cv2QPixmap(self, img):
        """
            将cv2图片转化为pixmap
        """
        if len(img.shape) == 3:
            pixmap = self.RgbCv2QPixmap(img)
        else:
            pixmap = self.GrayCv2QPixmap(img)
        return pixmap

    def showLoadImg(self, img):
        """
            图片显示到加载标签
        """
        pixmap = self.Cv2QPixmap(img)
        self.loadlabel.setPixmap(pixmap)

    @pyqtSlot()
    def cleartext(self):
        """
            清空输出文本框
        """
        self.textEdit.setText('')

    def loadimg(self):
        """
            加载图片
        """
        title = '选择图片'
        filter = '图片文件(*.jpg *.gif *.png *.bmp)'
        imgFileName = QFileDialog.getOpenFileName(self, title, self.cwd, filter)
        path = imgFileName[0]
        (self.cwd, filename) = os.path.split(path)
        if path != '':
            img = cv2.imread(path)
            img = self.reshapeImg(img)
            self.img = img
            self.showLoadImg(self.img)

    def ImgProcess(self):
        """
            处理图片
        """
        try:
            start = time.clock()
            gray = Process.Process_24CGM(self.img)
            end = time.clock()
            runtime = end - start
            pixmap = self.GrayCv2QPixmap(gray)
            self.showlabel.setPixmap(pixmap)
            self.textEdit.setPlainText('运行时间' + str(runtime) + '秒')
        except:
            self.textEdit.setPlainText('处理失败')


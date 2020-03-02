"""
    24_CGM处理测试GUI
    作者：张诺敏
    时间：2020.0301
"""
# coding=utf-8
import sys
from PyQt5.QtWidgets import QApplication
from mainWidgets import mainwidget

if __name__ == '__main__':

    app = QApplication(sys.argv)
    widget = mainwidget()
    widget.show()

    sys.exit(app.exec_())
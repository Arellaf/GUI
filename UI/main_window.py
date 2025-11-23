import os
from PyQt5.QtWidgets import QMainWindow
from main_UI import Ui_MainWindow
from UI.model1_page import Model1Page
from UI.model2_page import Model2Page
from UI.model3_page import Model3Page


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # --- Навігація між сторінками ---
        try:
            self.ui.icon_widget.hide()
        except Exception:
            pass
        try:
            self.ui.content.setCurrentIndex(0)
        except Exception:
            pass
        try:
            self.ui.lab1_btn_2.setChecked(True)
        except Exception:
            pass

        # --- Кнопки навігації ---
        try:
            self.ui.lab1_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
            self.ui.lab1_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(0))
            self.ui.lab2_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
            self.ui.lab2_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(1))
            self.ui.lab3_btn_1.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
            self.ui.lab3_btn_2.clicked.connect(lambda: self.ui.content.setCurrentIndex(2))
        except Exception:
            pass

        # --- Підключення сторінки з моделлю 1 ---
        self.regression_model = Model1Page(self.ui)

        # --- Підключення сторінки з моделлю 2 ---
        self.classification_model = Model2Page(self.ui)

        # --- Підключення сторінки з моделлю 3 ---
        self.regression_model = Model3Page(self.ui)
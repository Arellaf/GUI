import sys
from PyQt5.QtWidgets import QApplication
from UI.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        with open("style.qss", "r") as f:
            style_str = f.read()
            app.setStyleSheet(style_str)
    except FileNotFoundError:
        print("Файл style.qss не знайдено!")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

import sys
from PyQt5.QtWidgets import QApplication
import mainwindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainwindow.MainWindow()
    window.show()
    sys.exit(app.exec_())
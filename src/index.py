from main import Main
from ui import Ui

def start():
    ui = Ui()
    app = Main(ui)
    app.start()


if __name__=="__main__":
    start()

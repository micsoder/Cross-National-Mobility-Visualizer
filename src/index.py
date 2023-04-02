from main import Main
from ui import Ui
from input_packager import InputPackager

def index():
    ui = Ui()
    input_packager = InputPackager()
    app = Main(ui, input_packager)
    app.start()


if __name__=="__main__":
    index()

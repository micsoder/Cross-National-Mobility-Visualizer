from main import Main
from ui import ui
from input_packager import InputPackager

def index():
    input_packager = InputPackager()
    app = Main(ui, input_packager)
    app.main_loop()

if __name__=="__main__":
    index()

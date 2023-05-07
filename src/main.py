from kde_data_handler import KdeDataHandler
from country_codes import iso_country_codes
from BwAdjust import BwHandler

class Main():

    def __init__(self, ui, input_packager):
        self.ui = ui
        self.input_packager = input_packager
    
    def main_loop(self):

        while True: 

            state = self.ui.switch_state()

            if state == False:
                print('Program shutting down')
                break

            if state == "KDE":
                self.__initialize_kde()

            if state == "BW":
                self.__initialize_BW()


    def __initialize_kde(self):
            
        country1 = self.ui.kde_questions('first')
        country2 = self.ui.kde_questions('second')

        kde_hanlder = KdeDataHandler([country1, country2])
        kde_hanlder.start_processing()
        kde_hanlder.visualize()

    def __initialize_BW(self):

        Bw = self.ui.bw_handler_questions()

        if Bw == 'yes':
            bw_handler = BwHandler()
            print('test')
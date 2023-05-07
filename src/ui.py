from ui_state import UiState
from country_codes import iso_country_codes

class Ui():

    def __init__(self, state):
        self.state = state

    def switch_state(self):
        if self.state.children != None:
            self.state = self.state.state_engage()
        
        else:
            return self.state.title

    def kde_questions(self, country_number):

        while True:
            country = input(f'Add {country_number} country abbreviation: ')

            if country in iso_country_codes:
                print(f'{country} accepted')
                break
            print(f'{country} not accepted')
        return country


    def bw_handler_questions(self):

        question = input('Do you want to start the program: ')
        return question


# UiState def __init__(self, id, title, description, question):

state1 = UiState(1, 'Beginning state', 'In this program you can handle data and perform data analyses', 'What do you want to do? ')
state2 = UiState(2, 'Analysis', 'Here is a list of the analyses you can perform:', 'Which analysis do you want to perform? ')
state3 = UiState(3, 'KDE', 'Kernel Density Estimation analysis of two countries')
state4 = UiState(4, 'BW', 'Calculate the bandwidth of the entire dataset')

state1.add_children([state2, state4])
state2.add_parent(state1)
state2.add_children([state3])
state3.add_parent(state2)
state4.add_parent(state1)

ui = Ui(state1)






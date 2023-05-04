
class UiState():

    def __init__(self, id, title, state_question, parent=None, children=None):
        self.id = id
        self.title = title
        self.state_question = state_question
        self.parent = parent
        self.children = children


    def state_engage(self):

        print(self.state_question)
        self.options()

        self.input = input('What do you want to do?')
    
    def decision(self):
        if self.input == 'Exit':
            return False

        if self.parent != None:
            if self.input == 'Go back':
                return self.parent
        
        for child in self.children():
            if self.input == child.title:
                return child
            
        print('Incorrect input')
        return self
    
    def options(self):

        for child in self.children:
            print(child.title)
        
        if self.parent != None:
            print('Go back')
        
        print('Exit')


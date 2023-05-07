
class UiState():

    def __init__(self, id, title, description, question=None):
        self.id = id
        self.title = title
        self.description = description
        self.question = question
        self.parent = None
        self.children = None

    def add_parent(self, parent):

        self.parent = parent

    def add_children(self, children: list):

        self.children = children

    def state_engage(self):

        print(self.description)
        self.__options()

        self.input = input(self.question)

        return self.__decision()
    
    def __decision(self):
        if self.input == 'Exit':
            return False

        if self.parent != None:
            if self.input == 'Go back':
                return self.parent
        
        if self.children != None:
            for child in self.children:
                if self.input == child.title:
                    return child
                
        print('Incorrect input')
        return self
    
    def __options(self):
        
        if self.children != None:
            for child in self.children:
                print(child.title)
        
        if self.parent != None:
            print('Go back')
        
        if self.children == None:
            print(self.leaf_options)
        print('Exit')


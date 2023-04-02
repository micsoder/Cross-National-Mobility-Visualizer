
class Ui():

    def __init__(self):
        pass

    def title(self):
        print("Automated visualizastion of cross national border mobility")

 
    def question1(self):
        
        command = input("Add country (abbreviation) or exit program (exit): ")
        return command
     
    def question2(self):

        command = input("Add country (abbreviation) or exit program (exit): ")
        return command

    def question3(self):
        print("What do you want to do?")
        print("The available options are:")
        print("Kernel Density Estimation plotting (KDE)")

        command = input("Analysis: ")
        return command
    
    def start_program(self):
        
        command = input('Do you want to start the program (yes): ')
        return command
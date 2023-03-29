

class Main():

    def __init__(self, ui):
        self.ui = ui


    def start(self):
        
        while True:
            command = self.ui.question1()
            value = self.__take_command1(command)

            if value is False:
                print(f'Command "{command}" not recognized')
                continue
            
            break

        print(f"Command {command} accepted")
        self.second_question()

    def second_question(self):

        while True:
            command = self.ui.question2()
            value = self.__take_command2(command)

            if value is False:
                print(f'Second command "{command}" not recognized')
                continue

            break
                
        print(f"Command {command} accepted")
        print("Program ends here")

    def __take_command1(self, command):

        if command == 0:
            return True

        if command == 1:
            return True
        
        return False

    def __take_command2(self, command):

        if command == 0:
            return True

        if command == 1:
            return True

        return False
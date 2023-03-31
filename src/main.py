

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
        self.analysis_question()

    def analysis_question(self):

        while True:
            command = self.ui.question3()
            value = self.__take_command3(command)

            if value is False:
                print(f'Analysis "{command}" not recognized')
                continue
            break

    def __take_command1(self, command):

        if command == "exit":
            return False

        if command == 'FR':
            return True
        
        return False


    def __take_command2(self, command):

        if command == 'exit':
            return False

        if command == 'LU':
            return True

        return False

    def __take_command3(self, command):

        if command == 'exit':
            return False

        if command == 'KDE':
            return True

        return False
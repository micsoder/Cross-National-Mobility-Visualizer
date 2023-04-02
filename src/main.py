from kde_data_handler import KdeDataHandler

class Main():

    def __init__(self, ui, input_packager):
        self.ui = ui
        self.input_packager = input_packager


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
    
    def start_program_question(self):

        while True:
            command = self.ui.start_program()
            value = self.__take_command4(command)

            if value is False:
                print(f'Program not started')
                continue
            break


    def __take_command1(self, command):

        if command == "exit":
            return False

        if command == 'FR':
            self.input_packager.add_country(command)
            return True
        
        return False


    def __take_command2(self, command):

        if command == 'exit':
            return False

        if command == 'LU':
            self.input_packager.add_country(command)
            return True

        return False

    def __take_command3(self, command):

        if command == 'exit':
            return False

        if command == 'KDE':
            self.input_packager.add_analysis(command)
            return True

        return False
    
    def __take_command4(self, command):
        
        if command == 'yes':
            kde_handler = KdeDataHandler(self.input_packager.countries)
            kde_handler.start_analysis()
            kde_handler.visualize()
            return True
        
        else:
            return False
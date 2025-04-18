class IntroRenderer:
    def __init__(self):
        self.env_intros = {
            "default": self.get_intro,
            "tmaze": self.get_intro,
            "inference": self.get_intro_inference,
            "vizdoom": self.get_intro_vizdoom,
            "maniskill": self.get_intro_maniskill,
            "mem_maze": self.get_intro_mem_maze,
            "minigridmemory": self.get_intro_minigridmemory
        }

    def render_intro(self, env_name="default"):
        intro_func = self.env_intros.get(env_name, self.env_intros["default"])
        intro_func()

    def get_intro(self):
        print("⢸⠉⣹⠋⠉⢉⡟⢩⢋⠋⣽⡻⠭⢽⢉⠯⠭⠭⠭⢽⡍⢹⡍⠙⣯⠉⠉⠉⠉⠉⣿⢫⠉⠉⠉⢉⡟⠉⢿⢹⠉⢉⣉⢿⡝⡉⢩⢿⣻⢍⠉⠉⠩⢹⣟⡏⠉⠹⡉⢻⡍⡇")
        print("⢸⢠⢹⠀⠀⢸⠁⣼⠀⣼⡝⠀⠀⢸⠘⠀⠀⠀⠀⠈⢿⠀⡟⡄⠹⣣⠀⠀⠐⠀⢸⡘⡄⣤⠀⡼⠁⠀⢺⡘⠉⠀⠀⠀⠫⣪⣌⡌⢳⡻⣦⠀⠀⢃⡽⡼⡀⠀⢣⢸⠸⡇")
        print("⢸⡸⢸⠀⠀⣿⠀⣇⢠⡿⠀⠀⠀⠸⡇⠀⠀⠀⠀⠀⠘⢇⠸⠘⡀⠻⣇⠀⠀⠄⠀⡇⢣⢛⠀⡇⠀⠀⣸⠇⠀⠀⠀⠀⠀⠘⠄⢻⡀⠻⣻⣧⠀⠀⠃⢧⡇⠀⢸⢸⡇⡇")
        print("⢸⡇⢸⣠⠀⣿⢠⣿⡾⠁⠀⢀⡀⠤⢇⣀⣐⣀⠀⠤⢀⠈⠢⡡⡈⢦⡙⣷⡀⠀⠀⢿⠈⢻⣡⠁⠀⢀⠏⠀⠀⠀⢀⠀⠄⣀⣐⣀⣙⠢⡌⣻⣷⡀⢹⢸⡅⠀⢸⠸⡇⡇")
        print("⢸⡇⢸⣟⠀⢿⢸⡿⠀⣀⣶⣷⣾⡿⠿⣿⣿⣿⣿⣿⣶⣬⡀⠐⠰⣄⠙⠪⣻⣦⡀⠘⣧⠀⠙⠄⠀⠀⠀⠀⠀⣨⣴⣾⣿⠿⣿⣿⣿⣿⣿⣶⣯⣿⣼⢼⡇⠀⢸⡇⡇⠇")
        print("⢸⢧⠀⣿⡅⢸⣼⡷⣾⣿⡟⠋⣿⠓⢲⣿⣿⣿⡟⠙⣿⠛⢯⡳⡀⠈⠓⠄⡈⠚⠿⣧⣌⢧⠀⠀⠀⠀⠀⣠⣺⠟⢫⡿⠓⢺⣿⣿⣿⠏⠙⣏⠛⣿⣿⣾⡇⢀⡿⢠⠀⡇")
        print("⢸⢸⠀⢹⣷⡀⢿⡁⠀⠻⣇⠀⣇⠀⠘⣿⣿⡿⠁⠐⣉⡀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠳⠄⠀⠀⠀⠀⠋⠀⠘⡇⠀⠸⣿⣿⠟⠀⢈⣉⢠⡿⠁⣼⠁⣼⠃⣼⠀⡇")
        print("⢸⠸⣀⠈⣯⢳⡘⣇⠀⠀⠈⡂⣜⣆⡀⠀⠀⢀⣀⡴⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢽⣆⣀⠀⠀⠀⣀⣜⠕⡊⠀⣸⠇⣼⡟⢠⠏⠀⡇")
        print("⢸⠀⡟⠀⢸⡆⢹⡜⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠋⣾⡏⡇⡎⡇⠀⡇")
        print("⢸⠀⢃⡆⠀⢿⡄⠑⢽⣄⠀⠀⠀⢀⠂⠠⢁⠈⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠄⡐⢀⠂⠀⠀⣠⣮⡟⢹⣯⣸⣱⠁⠀⡇")
        print("⠈⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠁")
        
        print("___________                .__          ")
        print("\__    ___/_______ _____   |__|  ____   ")
        print("  |    |   \_  __ \\__  \  |  | /    \  ")
        print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
        print("  |____|    |__|   (____  /|__||___|  / ")
        print("                        \/          \/  ")

        

    def get_intro_inference(self):
        print(".___           _____                                                 ")  
        print("|   |  ____  _/ ____\  ____  _______   ____    ____    ____    ____  ")
        print("|   | /    \ \   __\ _/ __ \ \_  __ \_/ __ \  /    \ _/ ___\ _/ __ \ ")
        print("|   ||   |  \ |  |   \  ___/  |  | \/\  ___/ |   |  \\  \___ \  ___/ ")
        print("|___||___|  / |__|    \___  > |__|    \___  >|___|  / \___  > \___  >")
        print("          \/              \/              \/      \/      \/      \/ ")

        


    def get_intro_vizdoom(self):
        print(" =================     ===============     ===============   ========  ======== ")
        print(" \\ . . . . . . .\\   //. . . . . . .\\   //. . . . . . .\\  \\. . .\\// . . // ")
        print(" ||. . ._____. . .|| ||. . ._____. . .|| ||. . ._____. . .|| || . . .\/ . . .|| ")
        print(" || . .||   ||. . || || . .||   ||. . || || . .||   ||. . || ||. . . . . . . || ")
        print(" ||. . ||   || . .|| ||. . ||   || . .|| ||. . ||   || . .|| || . | . . . . .|| ")
        print(" || . .||   ||. _-|| ||-_ .||   ||. . || || . .||   ||. _-|| ||-_.|\ . . . . || ")
        print(" ||. . ||   ||-'  || ||  `-||   || . .|| ||. . ||   ||-'  || ||  `|\_ . .|. .|| ")
        print(" || . _||   ||    || ||    ||   ||_ . || || . _||   ||    || ||   |\ `-_/| . || ")
        print(" ||_-' ||  .|/    || ||    \|.  || `-_|| ||_-' ||  .|/    || ||   | \  / |-_.|| ")
        print(" ||    ||_-'      || ||      `-_||    || ||    ||_-'      || ||   | \  / |  `|| ")
        print(" ||    `'         || ||         `'    || ||    `'         || ||   | \  / |   || ")
        print(" ||            .===' `===.         .==='.`===.         .===' /==. |  \/  |   || ")
        print(" ||         .=='   \_|-_ `===. .==='   _|_   `===. .===' _-|/   `==  \/  |   || ")
        print(" ||      .=='    _-'    `-_  `='    _-'   `-_    `='  _-'   `-_  /|  \/  |   || ")
        print(" ||   .=='    _-'          `-__\._-'         `-_./__-'         `' |. /|  |   || ")
        print(" ||.=='    _-'                                                     `' |  /==.|| ")
        print(" =='    _-'                                                            \/   `== ")
        print(" \   _-'                                                                `-_   / ")
        print(" `''                                                                      ``'   ")

        print("___________                .__          ")
        print("\__    ___/_______ _____   |__|  ____   ")
        print("  |    |   \_  __ \\__  \  |  | /    \  ")
        print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
        print("  |____|    |__|   (____  /|__||___|  / ")
        print("                        \/          \/  ")

        

    def get_intro_maniskill(self):
        print(" ____    ____                _   ______   __       _  __  __      ")
        print(" |_   \\  /   _|              (_).' ____ \\ [  |  _  (_)[  |[  |  ")
        print(" |   \\/   |  ,--.  _ .--.  __ | (___ \\_| | | / ] __  | | | |    ")
        print(" | |\  /| | `'_\ :[ `.-. |[  | _.____`.  | '' < [  | | | | |      ")
        print(" _| |_\/_| |_// | |,| | | | | || \____) | | |`\\ \\ | | | | |     ")
        print(" |_____||_____\'-;__[___||__|___]\\______.'[__|  \\_|___|___|___] ")
        print("                                                                  ")

        print("___________                .__          ")
        print("\__    ___/_______ _____   |__|  ____   ")
        print("  |    |   \_  __ \\__  \  |  | /    \  ")
        print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
        print("  |____|    |__|   (____  /|__||___|  / ")
        print("                        \/          \/  ")

        


    def get_intro_mem_maze(self):
        print("  __  __                                 __  __                ")
        print(" |  \/  |                               |  \/  |               ")
        print(" | \  / | ___ _ __ ___   ___  _ __ _   _| \  / | __ _ _______  ")
        print(" | |\/| |/ _ \ ._ . _ \ / _ \| .__| | | | |\/| |/ _. |_  / _ \ ")
        print(" | |  | |  __/ | | | | | (_) | |  | |_| | |  | | (_| |/ /  __/ ")
        print(" |_|  |_|\___|_| |_| |_|\___/|_|   \__, |_|  |_|\__,_/___\___| ")
        print("                                    __/ |                      ")
        print("                                   |___/                       ")



        print("___________                .__          ")
        print("\__    ___/_______ _____   |__|  ____   ")
        print("  |    |   \_  __ \\__  \  |  | /    \  ")
        print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
        print("  |____|    |__|   (____  /|__||___|  / ")
        print("                        \/          \/  ")

        
                                                                                                                                                                                    


    def get_intro_minigridmemory(self):
        print("___________                .__          ")
        print("\__    ___/_______ _____   |__|  ____   ")
        print("  |    |   \_  __ \\__  \  |  | /    \  ")
        print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
        print("  |____|    |__|   (____  /|__||___|  / ")
        print("                        \/          \/  ")

        































# def get_intro_vizdoom():
#     print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⡶⠒⠛⠛⠛⠲⢦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠘⠻⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⣿⣿⢷⣤⣀⣀⣤⣦⣬⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣷⣟⣾⣿⡏⠁⠁⠁⠉⠛⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠶⠶⠶⠶⠶⠶⠦⠄⠤⠤⠄⠶⠶⠶⠶⠶⠶⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⣠⡤⠤⠤⠤⠤⠤⠤⠤⠤⠠⢤⡟⣸⣿⣿⢹⣿⣷⣤⣤⣤⡀⢘⣯⣷⠶⣞⠛⠛⠲⣤⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⣀⠀⠀⣠⣤⣤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⠀⠀⠀⠀⣠⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⣤⠀⠀⠉⢿⣿⣿⣿⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⣿⢻⣿⢛⣟⣿⠟⠛⠒⠂⠒⠒⠒⠒⠒⣒⣶⣾⣿⣯⣿⢿⣦⣀⠀⠀⠀⣠⡴⣋⣴⠞⢒⡒⠚⠷⣶⣖⡒⢲⣟⠁⣼⠿⣿⣼⣿⣿⣯⣽⣿⡿⠋⣽⢹⡄⠉⢷⡀⠀⠀⠙⢷⣶⠖⠶⠖⢶⡖⠒⠺⣧⡙⣷⣾⡁⠙⢻⣿⣷⣶⠶⣶⡶⢶⠶⣖⠲⣞⠻⡆⣀⡴⠟⢹⣿⠶⠶⢶⠶⠶⠶⢶⣶⣶⣦⣼⠟⠁⠀⠀⠀⠀⠈⢻⣿⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⣿⠿⠋⠁⠀⠀⡇⠀⡯⢠⣽⣿⡀⠀⠀⢀⣤⡰⠞⠉⢰⠟⠃⢀⡔⣤⠙⢷⣭⣿⣷⣄⡿⣿⡾⢋⠀⡀⣩⣅⡠⣴⠶⢿⣷⡌⠟⢷⣏⣸⣟⠹⣿⣧⣘⡿⠿⠷⣄⣿⡂⣻⠀⢸⡇⠀⠀⠀⠀⣙⣧⣠⡿⠇⢀⡀⣰⠞⠻⣷⡽⢿⡶⢻⠁⢸⠃⠀⠀⠀⠀⢀⡘⠃⢿⣷⡟⠋⠀⣼⡏⡿⠀⠀⠀⠀⢠⡾⠃⢸⣯⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿ ")
#     print("⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⡇⠀⡇⠀⢠⡟⢁⣤⣴⠛⠉⢁⡀⠀⠟⠀⠴⣫⠖⣣⣴⠛⣿⠿⣿⣟⡇⣿⡇⣸⠷⠳⠿⠻⢅⡤⢠⡄⡙⣦⡾⠳⣿⣿⣿⣦⣾⣿⠿⢧⣶⣶⣿⣿⡷⠛⠲⠾⣇⣀⣀⣀⡴⢋⣼⠟⠁⣰⣿⣾⣧⣤⣰⢻⣇⣼⣇⣼⡀⣸⣧⣠⣀⣀⡀⣈⠙⣤⡘⣇⣣⡀⢰⣿⣿⣇⠀⠀⠀⢠⠈⠀⢠⣽⣿⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿ ")
#     print("⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⡇⠀⣧⣴⠟⢾⣻⠈⢁⣀⣀⣀⣀⣀⣀⣀⡼⠇⡤⠛⣈⣻⡇⠀⣿⣿⡇⢸⣿⡇⠠⣼⡾⠃⢋⣀⡜⣻⣉⣩⣶⣤⢹⣿⣿⡿⢷⣿⣷⣿⣿⣿⠟⠋⠀⠀⠀⠀⢩⣿⣿⣿⣴⠟⠙⢷⣄⠐⢂⡀⠀⢉⣵⣿⡇⢸⡇⢸⠀⢸⡏⢻⣤⢠⡴⠳⡆⡼⠤⣿⡈⣧⡾⣿⡟⠈⠳⡔⠂⠀⠀⡀⠀⠀⡍⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿ ")
#     print("⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⡇⠀⡏⠋⢠⡾⠛⢀⣿⢻⣧⡤⢤⣼⡿⣿⠃⠠⠦⡾⣫⣿⡇⠀⣷⣼⡇⢸⡇⡇⢀⣬⢶⣦⢸⡏⣴⠟⠉⠉⠉⠉⠛⠻⣿⣷⣈⢉⣩⡿⠟⠋⣀⡴⢞⣤⠀⢀⡾⣱⣿⠛⠁⠀⠀⠈⢻⡇⠀⠉⠃⣿⠿⣿⡇⢸⠀⢸⠀⢾⡇⣤⣼⣣⣶⣴⢆⣶⠀⢈⣇⢹⢣⣿⠁⠀⡄⠐⡇⢀⣸⠁⢰⠄⡏⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿ ")
#     print("⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⡇⠀⡯⣾⠏⠀⣷⣼⣿⢸⡟⢳⣄⣠⡇⢿⠀⠀⣸⡏⣽⠁⣇⠀⣿⠃⡇⢸⡇⣹⣟⠁⠀⠛⣻⣿⡿⠛⠋⠉⠉⠙⠛⢶⣾⠏⠉⠉⠁⠀⠀⠉⢉⣵⡟⢁⡤⠈⠰⠋⣼⣷⣄⠀⠀⠀⠀⢻⣄⣀⠀⠁⠈⢹⡇⢸⡗⢸⠀⢻⢧⣶⠟⠛⠛⠛⠛⠛⠷⣄⣿⡄⣼⡟⢠⠼⣿⢠⡇⢸⡿⠀⠀⠰⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿ ")
#     print("⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠀⠀⠀⢱⡟⡏⢸⡟⢻⣿⣿⡇⣾⣠⡾⢋⡴⢿⣿⣿⢀⣿⢸⣧⣼⣿⣆⢋⡀⠀⢻⣿⡿⠆⠀⠀⠀⠀⠀⠀⠀⠙⣦⠀⠀⠀⠀⠀⠀⠀⣿⣿⠋⠀⠀⢀⣴⠃⠘⣿⣷⣤⣤⣄⡀⠈⠛⠶⢤⣄⣸⡇⢸⡅⢾⠀⣾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢷⣿⣦⠀⠀⠀⠀⠁⢸⣇⣀⢀⡀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿ ")
#     print("⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠀⢲⡄⢸⣹⡇⢸⣧⡀⢿⣿⡇⣿⢽⡏⣉⢶⢿⣿⣿⢾⡿⣌⣿⣿⣿⡿⣿⣷⣝⢻⣿⣸⡄⠀⠀⠀⠀⢀⣠⡤⠤⢼⣷⣄⠀⠀⠀⢀⣼⡿⠷⣤⣴⠾⣋⡀⠀⠀⢹⡍⣿⣿⣯⣷⠀⠰⢶⣿⣏⣽⡅⢸⡇⢸⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣶⣤⣄⠀⢧⣼⠇⡰⣿⠀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿ ")
#     print("⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⣀⢸⡄⢸⠉⡇⢸⣷⣿⡆⢸⡇⣸⢸⡏⠁⢈⣸⣿⣿⠘⣷⣾⡇⢻⣿⣿⡇⣉⣸⣿⣿⣯⠷⠀⠀⣠⢾⣫⠿⠛⠉⠙⠻⣿⣦⣶⣾⡿⠯⣴⣾⣿⡿⢿⣿⣿⣇⡀⠀⢻⡝⢿⣿⣿⣆⣠⣼⣿⣿⣞⣁⣈⣛⣾⣤⣿⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡝⠳⣄⣠⣿⣷⣿⣿⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹ ")
#     print("⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠉⠀⣀⢈⠛⡇⢨⠻⣯⣹⣾⡇⣿⠀⠀⣀⣼⣿⢿⡏⢸⣷⣿⣯⠈⡟⢿⣧⣻⣿⣿⣿⡿⣧⢀⡾⣣⡾⠁⠀⠦⡀⠀⠀⠈⢿⠉⢿⣷⠾⠛⣋⣁⣰⣰⣾⣯⠛⡟⢛⣿⣁⠀⠙⣿⣿⣿⣿⣿⠟⣻⣿⣿⣿⡿⠛⣿⠻⣤⣠⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⠙⣯⣈⣽⢿⣿⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣧⣴⠀⣼⣿⣤⣿⢸⣿⡏⣻⣿⡇⣿⣇⣼⡿⠋⠁⠈⡇⢸⡿⣿⡇⠀⣇⠀⠉⢻⣝⡻⣿⡇⣽⣿⣵⣿⣇⣴⣶⣆⠘⠦⠀⠀⠘⣇⣿⣥⣤⠶⠛⣩⣿⣼⣡⣾⠇⣡⣿⠟⠙⢷⣄⡏⣻⣟⣿⣿⣿⣿⡏⢹⣧⣤⣠⠾⣷⣾⣿⣿⣷⡤⣾⣧⠀⠀⢦⠀⠀⠀⠀⠀⠀⠉⠀⠘⣿⣣⣾⠁⡟⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡿⢻⣶⣟⠛⢻⡿⢸⢿⣿⣿⣾⡇⣿⠛⠿⠀⠠⠄⢀⡇⠸⣧⣸⡇⠀⡟⠀⠀⠀⠸⠿⢿⡇⢷⣟⢻⡟⣿⣿⣆⣸⣷⡀⠀⠀⠀⠹⣧⡭⣤⣶⣿⣿⡟⢛⣿⣷⡛⠉⠙⣷⣄⡼⣷⣟⠋⠁⣯⣼⣿⣓⡛⠀⣤⣬⠉⣰⣿⣿⣿⣿⠻⠇⢸⣿⠛⠳⠾⣦⡀⠀⠀⠀⠀⠀⠀⠀⠘⢻⣇⠀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⡅⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠘⠯⠋⠁⠈⡷⢸⠁⠙⢾⣿⡇⣿⠄⠀⠤⠤⠀⠨⡇⢀⣬⣧⡇⠀⡇⠀⠀⠀⠀⠀⢸⡇⣶⠇⢻⣿⣿⠙⣿⣿⣿⣿⠦⠀⠀⠀⠈⠻⣿⠷⣾⢿⣿⣿⣯⣽⣷⡶⠛⠉⣧⣴⠞⢹⡄⠀⠘⠘⢻⣍⡙⠻⣦⡉⠛⠛⢋⣿⣿⡿⢷⡆⠀⢻⠻⢦⣄⢻⣿⣷⣦⡀⠀⢀⣤⠞⢠⣿⣻⣤⡇⠀⡇⠀⠀⠀⠀⢀⣤⠄⠀⠀⠀⠀⣀ ")
#     print("⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠸⠐⢂⣤⣴⡇⢸⡐⠆⢈⢻⡇⠙⢀⣀⠀⠀⠀⠀⠇⠸⡏⢿⡇⠀⡇⠀⠀⠀⠀⢠⣸⡇⣩⡰⣮⠟⣿⠀⣿⢹⣿⣿⣷⣦⡀⠀⠀⠀⢹⣆⣿⣾⣿⣿⣿⣷⣌⠹⣷⠾⠿⠛⠃⠉⢳⣄⡀⣀⣈⣈⣿⣧⣄⡉⠳⠶⢿⡁⠀⠐⠺⠅⢈⠘⡇⠀⠙⢿⣿⣿⢿⣿⣆⡀⠀⠀⠸⣿⣿⣍⡇⠀⡇⠀⠀⠀⣴⢟⣁⣀⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣿⣤⠀⣀⣼⣿⡇⢸⠁⠳⣈⢿⡇⢀⠀⠀⣀⢘⣆⠀⠀⢠⡄⣿⡇⠀⡇⠀⠀⠀⠀⠀⢸⡇⢻⡷⣿⣦⣿⠀⣾⣹⠛⢿⣿⣿⣟⠳⣦⣴⣾⣿⣿⣿⣹⣿⣿⣿⣿⢿⡿⢦⣌⠀⢚⣓⡻⣇⠀⠂⠀⠀⣸⣿⡟⠻⣆⠰⣆⠙⣦⠀⠀⠀⢸⡀⢷⠀⠀⠀⠈⠻⣾⣷⣿⠟⣆⠉⠀⠈⠙⠿⢷⣄⡳⣤⡀⠘⠿⠛⠁⠀⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡷⢸⣶⡿⠿⠻⡇⢸⠀⠀⢩⣼⡇⣿⠀⠀⠿⠾⠿⠶⡆⠘⠷⡜⡇⠀⡇⠀⠀⠀⠀⠀⢸⡇⠀⠁⠈⠒⣿⣤⠟⠉⣀⣀⣙⣿⣿⣾⣿⣿⣿⣾⣟⣙⣿⣿⣿⣸⣿⣷⣻⣦⣭⣿⠉⠉⠉⣿⠀⠀⠀⢸⠍⢧⢻⡇⠘⣧⡘⣇⠸⡆⠀⠀⢸⣇⢸⡄⠂⠀⠀⠀⠀⢙⣿⣴⣙⡛⠓⢦⡁⠀⠀⠈⢿⣾⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⡆ ")
#     print("⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣷⠾⠿⠥⠀⠀⡇⢸⠀⢠⣾⡿⢃⡿⠄⢀⣀⡀⠀⠠⡇⢸⣧⣹⡇⠀⡇⠀⠀⠀⠀⠀⢸⣇⠻⣶⣶⡾⠋⠀⠀⣀⣿⡻⣟⣏⠙⣿⡿⣿⣿⠟⠙⢿⣿⣿⣷⣍⢿⣿⣿⣿⠛⠻⣦⡀⠀⢹⡖⠒⠒⠚⡅⢸⣦⠙⣆⡈⢷⣿⠀⢿⡀⠀⢸⣿⠀⠷⠄⠀⠈⠀⠀⣾⢻⡍⣿⢧⡀⠀⢳⣠⡄⠀⠀⢻⣟⢿⣿⣷⣦⡄⠀⠀⠀⠀⠀ ")
#     print("⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡇⠀⠀⠀⠀⡀⡇⢸⣰⠟⣫⡶⠋⠀⠀⠀⠻⠅⠀⠀⠀⠈⠉⠙⡇⠀⠃⠀⠀⢠⠀⠀⠀⠉⣳⣼⠟⠀⠀⣴⠟⢻⡌⣷⣿⣿⣀⣼⠿⡿⠃⠀⠀⠈⠙⢿⣿⣿⣿⣿⠿⣿⣿⣶⣄⠻⠞⠙⡇⠰⠀⠰⠃⢸⡏⣷⣜⣷⣸⣿⡾⠈⠻⢦⣼⣿⡇⠀⠀⠀⠠⠤⣤⡏⣼⠀⣿⢸⣧⡀⢨⣇⠻⠄⠀⠀⢙⣼⡷⠶⢿⡗⠀⠀⠀⠀⠀ ")
#     print("⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠇⠀⠀⠀⠚⠂⡇⢘⡵⠞⠛⠒⠒⠒⣆⠀⠀⠀⠀⠀⠀⢠⡆⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⢸⡏⠸⢷⠀⠀⠻⠟⠛⢻⡿⠿⠿⠛⠃⣀⢧⢠⡀⠀⠀⣠⣼⣿⣿⣿⣿⣦⢼⣿⢿⡟⠇⠀⠀⠀⠀⠀⠀⠀⠸⠟⠉⠋⣿⠏⠀⠙⢶⣄⠀⠙⣷⣿⠂⠀⠤⣄⣀⣿⢡⣿⠤⣿⣬⡿⣷⣶⡿⣷⣄⠰⣞⠋⠁⠀⣧⡀⣿⡀⠀⠀⢠⣦ ")
#     print("⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠖⠀⠀⠀⠀⠀⠿⠯⠄⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⡆⢸⣧⡁⡆⢀⠀⠀⠀⠀⠀⠀⠀⠸⣷⠀⢻⣷⡀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣤⣴⠟⠷⣤⢾⣯⢿⡽⣿⣧⡹⣇⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡀⢹⡠⢄⡀⠈⢿⣷⣄⢸⣿⡟⠳⣦⡀⢠⡇⣼⣽⣾⣿⣯⣽⣿⣿⣿⣾⣿⣿⣿⣧⣴⣿⣟⣷⣹⡇⠀⠀⣼⣿ ")
#     print("⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⡄⠀⠀⠀⠀⠀⠀⢐⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⠘⣿⣦⢷⣌⠳⢦⡀⠀⠀⠀⠀⠀⢻⣧⡀⢩⡗⠀⠀⡀⢰⣾⢿⡹⡾⠋⢯⠉⠷⣾⣿⣯⡿⣫⡿⢻⡏⢳⣽⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣏⡁⢨⣧⡈⠛⠂⠈⢻⡍⠁⡉⣷⠀⢸⡟⣶⡻⣿⠿⢽⣟⣿⣿⣿⣿⣿⣽⣿⣿⣿⣟⣷⣤⣈⣻⣿⠃⣀⠀⠉⠹ ")
#     print("⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡄⠀⠀⠰⠶⠈⣡⠤⣄⠉⠀⠀⢀⣤⠀⠀⠀⢀⡴⠋⢁⡴⠟⢻⡿⣿⣷⣤⡙⠶⣄⠀⠐⠒⠀⠙⣿⣉⠀⠀⠀⠁⠾⠋⠀⢳⣿⡀⢸⡇⠀⢹⣿⣳⡞⠛⢲⣺⣇⣘⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⡉⠘⠛⢦⣉⠳⣄⢻⣄⡀⠀⠀⢿⠈⣧⠀⠁⢸⡟⠁⢠⢿⣿⡿⠟⠛⢷⡙⠛⢿⣿⣿⡿⣋⣙⣿⡁⢹⠀⢹⡆⠀⠀ ")
#     print("⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣇⠀⠀⠀⠰⠶⢛⣛⣷⡦⣄⠀⠻⡄⠀⣠⠖⠋⣠⠶⠋⠀⠀⣰⡶⠻⣿⣿⡟⢦⣌⠳⢤⡀⠀⠀⠈⣻⣶⣄⠀⠀⠀⠀⠀⠈⣯⣧⣼⣿⡷⠶⠿⠋⠉⠳⣄⠙⢿⡏⠁⠀⡀⢠⠀⠀⠀⠀⠀⠀⠀⢠⣀⠀⠈⢻⠶⣄⠉⠳⣍⡳⢬⣙⣾⣿⡌⣧⠘⣧⠀⠘⢻⣶⣎⣾⣿⣀⣤⣾⣿⣷⣾⣾⣿⣿⡘⠿⣻⣿⣷⣾⠀⠘⠁⠀⠀ ")
#     print("⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⣇⣀⣯⡴⢶⣶⡀⢠⡟⣯⠈⠛⠏⠉⢀⡰⠛⢁⡴⠛⠁⠀⣶⣦⡞⠁⣀⣠⡾⠟⠛⢦⣌⢷⣤⡙⢦⣤⠛⢁⡽⢿⣦⡀⠀⠀⠀⢠⠹⣿⣧⣼⣷⡾⣿⠺⠟⠀⢈⡙⢦⣦⢴⡾⣿⣶⣾⠋⠀⣀⠀⡀⣠⡄⠸⣆⣷⣤⡀⢹⡇⣀⠈⠙⣦⡈⢻⢿⣇⣿⣧⡈⢷⡄⣮⣿⣿⣿⡿⣿⣿⣿⣿⡟⢛⣿⢿⡿⠛⢦⣌⡙⢿⣌⠶⠀⠀⠀⠀ ")
#     print("⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⣠⠿⠋⢀⣤⣾⣿⣿⣿⣿⣾⣧⣤⣠⣖⣋⣠⠾⠋⢀⣀⣀⣀⣈⣈⣙⠛⢿⣯⣥⣤⣤⣴⣿⡭⢾⣿⣿⠼⠿⠿⠤⠼⢿⣷⡄⠀⠀⣘⣧⣻⣿⣆⣼⣿⡿⠓⢶⡾⠃⢀⠀⢷⣶⡿⠟⣿⣿⣿⠘⣿⠀⣇⠀⣰⡟⠛⠛⢷⣍⣿⣦⡠⣿⣦⠀⣼⣿⢧⣿⡎⠉⢳⡄⢁⣿⣟⠛⣿⣿⣿⣿⣿⣷⣾⣿⣯⣏⠀⠀⢀⣿⡷⠶⠯⢭⣧⣄⠀⠀ ")
#     print("⣿⣿⣿⡟⠀⠀⠀⠀⣀⣴⠯⣤⣾⣿⣿⣿⣿⣿⣿⣯⣿⡏⠻⣥⡽⣿⣧⣤⣾⠛⠉⠀⠀⠙⣿⠟⠶⠶⠿⠻⠷⣦⠞⣁⣀⡀⣠⡴⠶⠟⠁⠀⣀⣠⠿⠿⡞⢋⣽⡿⣿⣿⣿⣷⣶⣿⣿⣋⠻⢦⣈⠀⣸⣿⠇⠐⣿⠿⠙⣆⡿⢸⣿⡞⠋⠀⠀⠀⠀⠉⠉⠉⠙⠾⠷⠞⠻⡇⢸⡏⢿⣄⡀⠙⠈⠿⣿⣶⣻⣿⣿⣿⠻⠛⠹⣿⣿⣿⣷⣾⠟⠁⠀⠀⠰⣤⠀⠉⠀⠀ ")
#     print("⣿⣿⣿⡟⠀⠀⢠⡞⢉⡀⣀⣀⢹⣿⣿⣿⢿⣿⣿⣿⣿⣁⡀⢻⣶⣿⣿⠿⢋⣤⡾⠃⠀⣼⠃⠀⣀⣤⠶⠶⠦⣴⣾⡟⠉⠉⠉⠀⢠⣾⢿⣿⣿⣇⣼⣿⣿⢿⣿⣹⣿⣷⡎⢿⣷⠚⠛⠋⠀⠀⠙⠿⢿⣿⡰⡀⠉⠀⠀⠀⠀⠀⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⢈⣧⡀⢹⡿⠦⠀⢸⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠉⠛⠶⣄⡈⠳⠂⠀⠀⠀⠀ ")
#     print("⣿⣿⣿⣿⡇⠀⢸⡄⠼⣿⣥⣴⡿⢋⣥⣶⣿⣿⣿⣿⣿⡿⣿⣄⢹⣭⣿⣶⡋⣁⣠⣤⣾⠇⠐⣩⡽⠖⠶⠖⠛⣿⡻⣷⡆⠀⠀⠀⣿⣿⣿⡉⢉⡿⠋⢉⣷⣿⢃⣿⣿⣿⣿⡟⣿⡀⠀⠀⠀⠀⠀⠀⠈⢿⣷⠇⠀⠀⠀⠀⠀⠀⣽⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠿⠋⢷⡀⠻⡄⠀⠈⠿⡏⣇⣿⠁⠀⠀⠀⠀⢀⣠⣀⠀⣀⠀⠀⠈⠟⠶⣤⡀⠀⠀⠀ ")
#     print("⣿⣿⣿⣿⣄⠀⠈⠻⣤⡿⣼⣿⣿⣿⣿⣛⡋⣹⣿⠟⢛⡶⣻⣿⡾⣿⣿⣫⣿⠿⠛⠉⣿⣀⣠⡤⠀⠀⠀⢰⣤⣌⣳⣾⣿⠶⠶⣶⠟⠛⢿⣿⣿⢷⣾⣻⡿⠃⣾⣿⣿⣿⣿⣿⣿⢿⣦⠀⠀⠀⠀⠀⠀⠈⢿⣧⠀⠀⠀⠀⠀⢠⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⡀⣬⠷⠶⢤⡀⠀⠀⢷⢺⡇⠀⠀⢀⣴⠾⠛⠉⠋⠀⠬⣍⡙⠳⢦⣀⣀⣙⣷⢶⣄ ")
#     print("⣿⣿⣿⣿⣿⣦⣠⣾⢋⣀⡛⠻⣿⣿⣿⣿⣷⣿⣿⣿⣤⣿⣿⣿⣿⡟⣩⡿⠃⠀⢠⡾⣿⣿⣅⣀⣀⣠⣤⠼⠟⠋⠁⠀⢀⣠⡼⠛⠀⠀⣿⠟⢃⣾⣿⣟⠻⣾⣿⣽⣿⢹⣟⣿⣿⣿⣿⣷⣤⣀⡀⠀⠀⠀⠘⣿⠄⠀⠀⠀⠀⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⢋⣭⣀⠉⠉⠋⠻⣦⠀⣻⡔⢦⣜⣸⠃⠀⣰⣿⡉⠛⠦⢤⣄⠈⠁⠀⠈⠁⣀⠸⠯⣭⣿⡮⣄ ")
#     print("⣿⣿⣿⣿⠿⣻⣿⠵⠞⠉⢿⠀⢀⣤⣀⣤⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠁⠀⠐⠋⠠⣿⣿⣿⣹⡏⢀⣴⠇⠀⢀⣤⣶⣿⡟⠁⣰⣿⣾⠃⢠⣿⣿⣿⣿⣼⡿⣿⠇⡿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⠀⠀⡇⠀⠀⠀⠀⠀⢹⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡴⠋⠁⠙⢷⢀⣶⣖⣿⡴⢾⣧⡀⠈⡏⠀⠀⢻⡟⠻⣷⣤⣄⣈⣳⣦⣤⣀⣀⡈⢹⣦⣼⡿⠃⠀ ")
#     print("⣿⣿⠏⠰⠾⠿⠒⣲⣦⣴⠟⠛⣿⣷⣝⣷⣶⣼⣿⠛⠛⠻⠿⢿⣟⣡⠞⠀⠀⠀⠀⠀⣿⡟⢸⡟⡗⠋⠁⣀⣴⣿⡿⢿⠏⢠⣾⣿⠞⢣⣴⣿⣿⣿⣿⣾⣿⣧⡈⣼⣷⢿⣿⣿⣿⣿⣹⣿⣽⣿⣿⡿⣿⢿⣦⣤⣄⣀⣤⣄⠀⠈⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⢴⡿⠎⢨⣯⣭⣿⠷⢾⣿⣾⣧⣴⣷⣴⣿⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀ ")
#     print("⡟⠁⠀⢠⣴⣞⣛⡿⠏⠻⣷⣶⡊⢻⣿⠯⣿⣿⣾⣇⣀⠀⠀⣿⣋⡁⢀⣠⠶⠀⢀⣰⠟⣀⣸⣧⣧⣴⣞⣋⣟⡋⠀⢀⣰⣿⠟⢁⡀⠂⣸⠟⠿⠛⢻⣿⠿⣯⣿⠇⣿⣾⣿⢿⡿⢿⣏⣿⣿⣿⣿⡇⠙⣷⣋⣵⡿⣿⣦⡜⣧⠀⠿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣁⣈⣿⡾⠋⠙⣿⣿⣿⣿⣯⣽⠏⠉⠉⠹⠏⠉⠉⠉⠟⠻⡏⠉⠉⠉⠉⠉⠉⠙⠻⠇⠀⠀ ")

#     print("___________                .__          ")
#     print("\__    ___/_______ _____   |__|  ____   ")
#     print("  |    |   \_  __ \\__  \  |  | /    \  ")
#     print("  |    |    |  | \/ / __ \_|  ||   |  \ ")
#     print("  |____|    |__|   (____  /|__||___|  / ")
#     print("                        \/          \/  ")
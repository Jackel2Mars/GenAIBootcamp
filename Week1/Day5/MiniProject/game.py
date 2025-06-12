import random

class Game():
    def __init__(self):
        #values for user input, along with their corresponding string for further displays
        self.accepted_values = {"r": "rock", "p": "paper", "s": "scissors"}

        #look_up for results
        self.result_lookup = {
            ("r", "r"): "draw",
            ("r", "s"): "win",
            ("r", "p"): "loss",
            ("s", "r"): "loss",
            ("s", "s"): "draw",
            ("s", "p"): "win",
            ("p", "r"): "win",
            ("p", "s"): "loss",
            ("p", "p"): "draw"
        }
        #they were put in the contructor in order to:
        #1. Have only one place to update
        #2. Be instanciated only one per game

    def get_user_item(self):
        user_value = "" #initiate user_value as empty string in order to enter the loop
        while user_value not in self.accepted_values:
            user_value = input("Select (r)ock, (p)aper, or (s)cissors: ").strip().lower() #get rid of whitespaces and enable case flexibility
            if user_value not in self.accepted_values:
                print("Invalid input. Please enter 'r', 'p', or 's'.")
        return user_value

    def get_computer_item(self):
        return random.choice(list(self.accepted_values.keys()))

    def get_result(self, user_item, computer_item):
        if user_item not in self.accepted_values or computer_item not in self.accepted_values:
            return "Invalid input" #returns "Invalid input" if user_item or computer_item are not "r", "s" or "p". 
        return self.result_lookup.get((user_item, computer_item), "Invalid input") # returns the value of result_lookup dictionary for (user_item, computer_item) key couple if it exists. Returns "Invalid input" if key couple (user_ietm, computer_item) is not found in dictionary. 


    def play(self):
        #request user_item from user
        user_item = self.get_user_item()
        #have computer generate a value for computer_ietm
        computer_item = self.get_computer_item()
        #retrieve result of the rock-paper-scissor game
        result = self.get_result(user_item, computer_item)
        #contruction of the string to print in order to display result.
        result_string = f"You selected {self.accepted_values[user_item]}. The computer selected {self.accepted_values[computer_item]}. " #string initialization
        #concatenation of other strings depending on the value of result
        if result == "loss":
            result_string += "You lost!"
        elif result == "win":
            result_string += "You won!"
        else:
            result_string += "You drew!"
        print(result_string) #display of result
        return result



    


        





from game import Game

def get_user_menu_choice():
    #retrive user choice from game menu
    user_menu_choice = input("""Menu
    (g) Play a new game
    (x) Show scores
    (q) Quit
    : """).strip().lower() #get rid of whitespaces and enable case flexibility
    return user_menu_choice

def print_results(results):
    print(f"""Game Results:
    You won {results["win"]} times
    You lost {results["loss"]} times
    You drew {results["draw"]} times""")

def main():
    results = {
        "win": 0,
        "loss": 0,
        "draw": 0
    } #initialization of results dictionnary
    #loop until user descides to quit
    while True:
        user_menu_choice = get_user_menu_choice() #stores user choice from game menu in a variable
        if user_menu_choice == "q":
            print_results(results)
            break #quit loop
        elif user_menu_choice == "g":
            game = Game() #instanciation of Game
            result = game.play() #launches the new game and stores the result in a variable
            if result in results:
                results[result] += 1 #incrementation of results dictionary depending on value of result
        elif user_menu_choice == "x":
            print_results(results)
        else:
            print("Invalid menu choice. Please choose (g), (x), or (q).") #Error message if the value for user choice is not one we requested.

main()
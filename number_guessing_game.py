import random

def number_guessing_game():
    random_number = random.randint(1, 100)
    max_attempts = 7
    for attempt in range(max_attempts):
        guess = int(input("Guess an integer number between 1 and 100!")) #since input returns a string we cast it as integer
        #Comparison between the number retrieved from user, guess, and the number generated randomly, random_number.
        if guess < random_number:
            print("Too low!")
        elif guess > random_number:
            print("Too high!")
        else:
            print(f"Congratulations on finding {random_number}")
            break
    if guess != random_number:
        print(f"The number was {random_number}") #Display of a message if the number wasn't found through all attempts

number_guessing_game() #call for number_guessing_game function
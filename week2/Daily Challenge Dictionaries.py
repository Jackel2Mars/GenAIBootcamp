"""Daily Challenge: Dictionaries
Challenge
    Ask a user for a word.
    Write a program that creates a dictionary. This dictionary stores the indexes of each letter in a list.
    Make sure the letters are the keys.
    Make sure the letters are strings.
    Make sure the indexes are stored in a list, and those lists are values.
Examples
    “dodo” ➞ { “d”: [0, 2], “o”: [1, 3] }
    “froggy” ➞ { “f”: [0], “r”: [1], “o”: [2], “g”: [3, 4], “y”: [5] }
    “grapes” ➞ { “g”: [0], “r”: [1], “a”: [2], “p”: [3] }"""
word = input("write down a word. ")
letters_vs_indexes = {}
for index, letter in enumerate(word):
    if letter not in letters_vs_indexes:
        letters_vs_indexes[letter] = []
    letters_vs_indexes[letter].append(index)
    
print(f'\n"{word}" ➞ {{')
for letter, indexes in letters_vs_indexes.items():
    print(f'  "{letter}": {indexes}')
print('}')


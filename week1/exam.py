from utils import unzip_with_7z

zip_file_path = 'congrats.7z' # keep as is
dest_path = '.' # keep as is

find_me = '' # 2 letters are missing!
secret_password = find_me + 'bcmpda' 

# WRITE YOUR CODE BELOW
# ----------------------------------------
def try_password():
    import string
    alphabet = string.ascii_lowercase #retrieve all lower case letters
    #loop in order to generate a combination of lower case letters
    for first_letter in alphabet:
        for second_letter in alphabet:
            find_me = first_letter + second_letter #concatanation of two lower case letters
            secret_password = find_me + 'bcmpda' #concatenation of the lower case letters with the known part of the password
            attempt = unzip_with_7z(zip_file_path, dest_path, secret_password) #password gets sumbmitted
            if attempt == True:
                return #added in order to stop the process if the password is found before reaching the end of the loops

        
try_password() #call for try_password method

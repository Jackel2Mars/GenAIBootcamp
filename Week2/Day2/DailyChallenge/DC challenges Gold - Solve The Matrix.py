"""Given a “Matrix” string:

7ii
Tsx
h%?
i #
sM 
$a 
#t%
^r!


The matrix is a grid of strings (alphanumeric characters and spaces) with a hidden message in it.
A grid means that you could potentially break it into rows and columns, like here:

7	i	i
T	s	x
h	%	?
i		#
s	M	
$	a	
#	t	%
^	r	!


Matrix: A matrix is a two-dimensional array. It is a grid of numbers arranged in rows and columns.
To reproduce the grid, the matrix should be a 2D list, not a string



To decrypt the matrix, Neo reads each column from top to bottom, starting from the leftmost column, selecting only the alpha characters and connecting them. Then he replaces every group of symbols between two alpha characters by a space.

Using his technique, try to decode this matrix.

Hints:

Use

● lists for storing data
● Loops for going through the data
● if/else statements to check the data
● String for the output of the secret message

Hint (if needed) : Look at the remote learning “Matrix” video."""

matrix_string = """7ii
Tsx
h%?
i #
sM 
$a 
#t%
^r!"""

rows = matrix_string.split('\n')
matrix = [list(row) for row in rows]
num_rows = len(matrix)
num_cols = len(matrix[0])  # assuming all rows have same length

read_chars = []
for col in range(num_cols):
    for row in range(num_rows):
        read_chars.append(matrix[row][col])

message_raw = ''.join(read_chars)
cleaned = ""
i = 0
length = len(message_raw)

while i < length:
    if message_raw[i].isalpha():
        cleaned += message_raw[i]
        i += 1
    else:
        # We hit a symbol — check if it's between letters
        # Add a space if next alpha comes after this junk
        cleaned += " "
        # Skip all consecutive non-alpha
        while i < length and not message_raw[i].isalpha():
            i += 1
print(cleaned)

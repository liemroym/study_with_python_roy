import random # Library for randomizer
import time # Libary for time (delaying program for a while)

field = [['.', '.', '.'], # Initialize empty game field
         ['.', '.', '.'], 
         ['.', '.', '.']]
    
def Start():
    print(
f'''======================TIC TAC TOE=======================
Game Description: 
You're playing as X! Connect 3 before your opponent does!

How to play: 
Input where you want to insert your X's by typing in the 
row and col values. For example, typing in 3 and 2 would 
insert an X at 3rd row, 2nd column.
'''
    )
    time.sleep(2) # Delay for 2 second
    InitField()
    RollChance()
    
def InitField():
    for row in field:
        print(" ".join(str(cell) for cell in row))

def RollChance():
    chance = random.randint(0, 1)
    if chance == 0:
        print("\nThe player starts first!")
        time.sleep(1)
        Play()
    else:
        print("\nThe bot starts first!")
        time.sleep(1)
        BotFill()
        time.sleep(2)
        Play()

def Play(): # Start playing the game
    gameFinished = False
    print("\nPlease input where you want to insert an X")
    
    while True:
        try:
            row = int(input("row: "))
            col = int(input("col: "))
        except ValueError:
            time.sleep(1)
            print("\nNot a valid value (Please input an integer between 1 to 3)")
            print(*field, sep='\n')
            time.sleep(1)
            continue

        if row > 3 or col > 3 or row < 1 or col < 1: # Check if the user input is valid or not (1 <= row/col <= 3)
            time.sleep(1)
            print("\nNot a valid value (Out of range, please input an integer between 1 to 3)")
            print(*field, sep='\n')
            time.sleep(1)
            continue
        else:
            break

    InsertX(row, col) # Insert an X according to user input
    
    if WinCondition('X'):
        print("Player Wins")
        gameFinished = True
        time.sleep(2)
        End()
    
    if IsFull() and gameFinished == False: # Check if the field is full or not. End the game if full
        print("The field is full")
        gameFinished = True
        time.sleep(2)
        End() 
    elif gameFinished == False:
        BotFill() # Fill an O in random box

    if WinCondition('O') and gameFinished == False:
        print("Player Loses")
        gameFinished = True
        time.sleep(2)
        End() 
    
    if IsFull() and gameFinished == False: # Recheck after the bot filled a box (useful when the one who's starting the game is randomized)
        print("The field is full")
        gameFinished = True
        time.sleep(2)
        End()
    elif gameFinished == False:
        time.sleep(2)
        Play() # Call the Play() function again until eventually the field is full

def IsFilled(row, col): # Check whether a box is filled or not. If filled --> True
    return field[row][col] != '.'

def InsertX(row, col): # Insert an X according to user input
    if IsFilled(row-1, col-1): # Check if the box that the user inputs is filled or not
        print("\nNot a valid value (Filled)")
        time.sleep(1)
        InitField()
        Play()

    else: # Input is correct and box is fillable
        field[row-1][col-1] = 'X'
        print("\nYou filled an X at ({}, {})".format(row, col))
        InitField()

def BotFill(): # Function to fill a random box with an O
    row = random.randint(1, 3) # Produce a random integer from 1 to 3 as the row where the O will be inserted 
    col = random.randint(1, 3) # Produce a random integer from 1 to 3 as the collumn where the O will be inserted
    
    if IsFilled(row-1, col-1): # Check whether the box is filled or not. If filled, call BotFill() function again to get a new random row and col
        BotFill() 
    else: # Not filled. Insert an O to the box
        field[row-1][col-1] = 'O'
        time.sleep(2)
        print("\nThe bot filled an O at ({}, {})".format(row, col))
        InitField()

def IsFull(): # Check whether the field is full or not
    value = True # Initial value, just a variable to make this function easier to process
    for i in range(3):
        for j in range(3):
            if IsFilled(i, j): # Check whether filled 
                value = value and True 
            else: # Or not
                value = value and False
    return value

def WinCondition(xo):
    if ((field[0][0] == xo and field[1][1] == xo and field[2][2] == xo) or (field[0][2] == xo and field[1][1] == xo and field[2][0] == xo)):
        return True
    for i in range(3):
        if ((field[i][0] == xo and field[i][1] == xo and field[i][2] == xo) or (field[0][i] == xo and field[1][i] == xo and field[2][i] == xo)):
            return True
    else:
        return False              

def End(): # End the game
    yn = str(input("\nDo you want to play again? Yes/No: "))

    if yn == "Yes":
        print("\n")
        global field
        field = [['.', '.', '.'], # Initialize empty game field
                 ['.', '.', '.'], 
                 ['.', '.', '.']]
        Start()
    elif yn == "No":
        print("\nSee you next time!")
    else:
        print("Not a valid answer (Please type in 'Yes' or 'No'!)")
        End()

Start()

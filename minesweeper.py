import random
import time

def Start():
    print(
'''
=====================MINESWEEPER=====================
Welcome to minesweeper! Your goal is to open all the 
boxes while avoiding the mines. First, choose your 
desired difficulty:
1. Easy     (10x10, 12 bombs)
2. Normal   (15x15, 22 bombs)
3. Hard     (20x20, 30 bombs)
''')
        
    PickDiff()
    CheckDiff()
    PlayerInput()

def PickDiff():
    while True:
        try:
            global diff
            diff = int(input("Choose your difficulty (Input the number): "))    
        except ValueError:
            time.sleep(1)
            print("\nPlease input a value between 1 to 3\n")
            continue

        if diff < 1 or diff > 3:
            time.sleep(1)
            print("\nPlease input a value between 1 to 3\n")
            continue
        else:
            break

def CheckDiff():
    global n
    if diff == 1:
        print("\nDifficulty selected: Easy\n")
        time.sleep(1)
        n = 10
        GameField(12)
    elif diff == 2:
        print("\nDifficulty selected: Medium\n")
        time.sleep(1)
        n = 15
        GameField(22)
    else:
        print("\nDifficulty selected: Hard\n")
        time.sleep(1)
        n = 20
        GameField(30)

def MakeField(char):
    arr = [[char for row in range(n)] for column in range(n)]
    return arr

def GameField(b):
    global realField
    global playField
    realField = MakeField(0)
    playField = MakeField('n')

    for i in range(b):
        row = random.randint(0, n-1)
        col = random.randint(0, n-1)

        PlaceBomb(row, col)

def InitField(field):
    for row in field:
        print(" ".join(str(cell) for cell in row))
        
def PlaceBomb(row, col):
    if realField[row][col] != 'X':
        realField[row][col] = 'X'        

        if row > 0:
            if realField[row-1][col] != 'X':
                realField[row-1][col] += 1
        if row < n-1:
            if realField[row+1][col] != 'X':
                realField[row+1][col] += 1
        if col > 0:
            if realField[row][col-1] != 'X':
                realField[row][col-1] += 1
        if col < n-1:
            if realField[row][col+1] != 'X':
                realField[row][col+1] += 1
        if row > 0 and col > 0:
            if realField[row-1][col-1] != 'X':
                realField[row-1][col-1] += 1
        if row < n-1 and col > 0:
            if realField[row+1][col-1] != 'X':
                realField[row+1][col-1] += 1
        if row > 0 and col < n-1:
            if realField[row-1][col+1] != 'X':
                realField[row-1][col+1] += 1
        if row < n-1 and col < n-1:
            if realField[row+1][col+1] != 'X':
                realField[row+1][col+1] += 1    
    else:
            newRow = random.randint(0, n-1)
            newCol = random.randint(0, n-1)
            PlaceBomb(newRow, newCol)

def PlayerInput(method = None):
    global gameEnd, inputMethod
    gameEnd = False

    if method == None:
        inputMethod = "OPEN"
    else:
        inputMethod = method

    InitField(playField)
    time.sleep(1)
    
    print("\nInput the row and column of the cell you desire. Input 0 in any input to change your input method (flag or open cell)")
    while True:
        try:
            rowInput = int(input("\nInput the row of the cell: "))
            colInput = int(input("Input the column of the cell: "))
        except ValueError:
            time.sleep(1)
            print("\nPlease input an integer\n")
            time.sleep(1)
            InitField(playField)
            continue
        
        if rowInput == 0 or colInput == 0:
            time.sleep(1)
            print("\nChange the input method?\n1. Yes       2. No\n")
            
            while True:
                try:
                    ans = int(input("Your answer: "))
                except ValueError:
                    print("\nPlease input 1 or 2\n")
                    continue
                
                if ans == 1:
                    if inputMethod == "OPEN":    
                        inputMethod = "FLAG"
                        print("\nInput method changed to: flag\n")
                        PlayerInput(inputMethod)
                        break
                    else:
                        inputMethod = "OPEN"
                        print("\nInput method changed to: open cell\n")
                        PlayerInput(inputMethod)
                        break
                elif ans == 2:
                    PlayerInput(inputMethod)
                    break
                else:
                    print("Please input 1 or 2")
                    continue 

        elif rowInput < 1 or rowInput > n or colInput < 1 or colInput > n:
            time.sleep(1)
            print("\nNot a valid value (Please input between 1 to {})\n".format(n))
            time.sleep(1)
            InitField(playField)
        else: 
            break
    
    print("\nInputted cell: Cell[{}][{}]".format(rowInput, colInput))
    CheckInput(rowInput-1, colInput-1)

    if WinCondition() == True:
        gameEnd = True
        InitField(realField)
        print("\nYou Win!\n")
        time.sleep(1)
        Restart()
    
    if gameEnd == False:
        PlayerInput(inputMethod)    

def CheckInput(row, col):
    global inputMethod
    if inputMethod == "OPEN":
        CheckCell(row, col)
    else:
        if playField[row][col] == 'F':
            playField[row][col] = 'n'               
        elif playField[row][col] == 'n':
            playField[row][col] = 'F'
        else:
            FlagAround(row, col)

def CheckCell(row, col):
    global gameEnd
    if playField[row][col] != 'n':
        FlagAround(row, col)
    elif realField[row][col] == 'X' and gameEnd == False:
        gameEnd = True
        realField[row][col] = 'L'
        InitField(realField)
        print("\nYou Lose!\n")
        time.sleep(1)
        Restart()
    elif realField[row][col] == 0:
        playField[row][col] = 0
        CellAround(row, col, CheckCell)
    elif playField[row][col] != 'F':
        playField[row][col] = realField[row][col]

def FlagAround(row, col):
    flagCount = 0

    if row > 0:
        if playField[row-1][col] == 'F':
            flagCount += 1
    if row < n-1:
        if playField[row+1][col] == 'F':
            flagCount += 1
    if col > 0:
        if playField[row][col-1] == 'F':
            flagCount += 1
    if col < n-1:
        if playField[row][col+1] == 'F':
            flagCount += 1
    if row > 0 and col > 0:
        if playField[row-1][col-1] == 'F':
            flagCount += 1
    if row < n-1 and col > 0:
        if playField[row+1][col-1] == 'F':
            flagCount += 1
    if row > 0 and col < n-1:
        if playField[row-1][col+1] == 'F':
            flagCount += 1
    if row < n-1 and col < n-1:
        if playField[row+1][col+1] == 'F':
            flagCount += 1
        
    if flagCount >= playField[row][col]:
        CellAround(row, col, CheckCell)
    else:
        print("Insuficient amount of flag around cell[{}][{}]".format(row+1, col+1))

def CellAround(row, col, func):
    if row > 0 and playField[row-1][col] == 'n':
        func(row-1, col)
    if row < n-1 and playField[row+1][col] == 'n':
        func(row+1, col)
    if col > 0 and playField[row][col-1] == 'n':
        func(row, col-1)
    if col < n-1 and playField[row][col+1] == 'n':
        func(row, col+1)
    if row > 0 and col > 0 and playField[row-1][col-1] == 'n':
        func(row-1, col-1)
    if row < n-1 and col > 0 and playField[row+1][col-1] == 'n':
        func(row+1, col-1)
    if row > 0 and col < n-1 and playField[row-1][col+1] == 'n':
        func(row-1, col+1)
    if row < n-1 and col < n-1 and playField[row+1][col+1] == 'n':
        func(row+1, col+1)
        
def WinCondition():
    win = True
    for row in range(n):
        for col in range(n):
            if (playField[row][col] == 'n' or playField[row][col] == 'F') and realField[row][col] != 'X':
                win = win and False
    return win

def Restart():
    print("Do you want to play again?\n 1. Yes      2. No")
    while True:
        try:
            yn = int(input("\nInput: "))
        except ValueError:
            time.sleep(1)
            print("\nInvalid input (Input 1 or 2)\n")
            time.sleep(1)
            continue
    
        if yn == 1:
            time.sleep(1)
            Start()
            break
        else:
            time.sleep(1)
            print("See you next time! (^o^)/")
            break
Start()
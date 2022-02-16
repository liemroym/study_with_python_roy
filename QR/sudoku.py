field = [['.' for row in range(9)] for col in range(9)]

def InitField():
    for row in field:
        for i in range(0, 9, 3):
            print('j'.join(str(row)))

InitField()
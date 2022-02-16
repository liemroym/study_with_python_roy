# IsPrime: int --> boolean, 2 int
#   IsPrime() menentukan apakah suatu integer merupakan bilangan prima atau bukan. Jika bukan, maka diberikan pula faktor pembentuk nilai tersebut selain 1 kali angka itu sendiri
def IsPrime():
    
    n = int(input("Input your number: "))
    prime = True
    if n > 1:
        for i in range(2, n):
                if n % i == 0:
                    prime = False 
                    print(i, "multiplied by", n//i, "is equal to", n)       
        
        if prime == True:
            print("{} is a prime number".format(n))
        else:
            print("{} is not a prime number".format(n))
        
    else:
        return ("Number 1 is not a prime number")

def ClosestPrime(n = None):
    prime = True
    
    if n == None:
        n = int(input("Input your number: "))
    if n > 1:
        for i in range(2, n):
                if n % i == 0:
                    prime = False

    if prime == False:
        print("{} is not a prime number".format(n))
        ClosestPrime(n - 1)
    else:
        print("{} is a prime number".format(n))    

#IsPrime()
#ClosestPrime()

print(4**18 * 19**80 % 9)



# import re

# pattern = "(\d\d)-(\d\d\d)-(\d\d\d\d)"

# pattern2 = r"\1\2\3"

# number = input()

# number2 = re.sub(pattern, pattern2, number)

# print(number2)

# x1 = 7
# y1 = 10
# x2 = 17
# y2 = 16

# dx = abs(x1-x2)
# dy = abs(y1-y2)

# p = 2*dy - dx

# while (x1 != x2 or y1 != y2):
#     print('p =', p, ', x =', x1, ', y =', y1)
#     x1 += 1
#     if (p <= 0):
#         p += (2*dy)
#     else:
#         y1 += 1
#         p += (2*dy - 2*dx)

# print(x2, y2)    

# x0, y0 = 3, 4
# r = 10


# p = 1-r

# x = 0 
# y = r

# print ("Awal: ", x0, y0+r)
# while (y > x):
#     print("p = ", p)
#     x += 1
#     if (p < 0):
#         p = p + 2*(x+1) + 1
#     else:
#         p = p + 2*(x+1) + 1 - 2*(y-1)
#         y -= 1

#     print("x, y =", x+x0, y+y0)

# arr = [2, 7, 11, 15]
# x = 9

arr = [5, 1, 4, 9, 7, 5, 6]
x = 10
def find_pair(arr, x):
    for i in range(len(arr)): # n
        for j in range(i+1, len(arr)): # (n(n+1) / 2) - n
            if (arr[i] + arr[j] == x): 
                return ((arr[i], arr[j]))

    return 0

def find_pair_2(a, n, x):
     
    rem = []
     
    for i in range(x):
 
        # Initializing the rem
        # values with 0's.
        rem.append(0)
 
    for i in range(n):
        if (a[i] < x):
 
            # Perform the remainder operation
            # only if the element is x, as
            # numbers greater than x can't
            # be used to get a sum x.Updating
            # the count of remainders.
            rem[a[i] % x] += 1
 
    # Traversing the remainder list from
    # start to middle to find pairs
    for i in range(1, x // 2):
        if (rem[i] > 0 and rem[x - i] > 0):
 
            # The elements with remainders
            # i and x-i will result to a
            # sum of x. Once we get two
            # elements which add up to x,
            # we print x and break.
            count += 1
 
    # Once we reach middle of
    # remainder array, we have to
    # do operations based on x.
    if (i >= x // 2):
        if (x % 2 == 0):
            if (rem[x // 2] > 1):
 
                # If x is even and we have more
                # than 1 elements with remainder
                # x/2, then we will have two
                # distinct elements which add up
                # to x. if we dont have than 1
                # element, print "No".
                count += 1
        else:
 
            # When x is odd we continue
            # the same process which we
            # did in previous loop.
            if (rem[x // 2] > 0 and
                rem[x - x // 2] > 0):
                count += 1



print(find_pair(arr, x))
print(find_pair_2(arr, x))
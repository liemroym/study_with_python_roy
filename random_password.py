import string
from random import *
characters = string.ascii_letters + string.punctuation  + string.digits

for i in range(45):
    password =  "".join(choice(characters) for x in range(60))
    print (password)

print(len("Software Engineering"))

# hai = f'''
# Because we are a fashion driven field. That’s it. OOP(the kind of OO in java, C++ etc) is not better than functional or logic or even procedural programming it just won the popularity contest, like windows or linux or C.

# Software “engineering” is full of people that have no idea what came before them. They have no idea what is possible and just go with the herd. Just ask any of them to name 5 turing award winners and their contributions to the field. And so it is a field destined to not improve itself in any substantial way.

# Universities are just worried about pumping graduates that know what the industry wants and the industry wants uniformity and conformity and whatever they are already using. And so software “engineering” is destined to keep using the same crap for the next 10 or 20 years without any regards for the artifacts it produces.

# Who cares about the users? The hardware is a million times faster and yet the software keeps getting slower, uses more memory, the usability is terrible but hey! We sure know that OOP is awesome. And now we want OOP in the large AKA microservices AKA how to make a system many times slower by introducing a network operation between every module. But hey at least we have chat applications that only consume 20% CPU of a 4 core machine and only uses 1gb to display text and some pictures. Great! Long live OOP!'''

# hai = hai.replace('\n', '@')
# print(hai)



import random

for i in range(50):
    koef = []
    degree = random.randint(2, 10)
    for j in range(degree):
        koef.append(random.uniform(0, 20))
        
    h = random.uniform(0, 0.5)
    x0 = random.uniform(-20, 20)
    print(koef)
    f = open('data.txt', 'a')
    for item in koef:
        f.write("%s " % item)
    f.write("%s " % random.uniform(-19, 19))
    f.write("%s " % h)
    f.write("%s " % x0)
    f.write("%s\n" % degree)
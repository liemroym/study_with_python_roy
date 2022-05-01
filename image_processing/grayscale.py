import matplotlib.image as image
from PIL import Image
import numpy as np

matrix=image.imread('D:\\1. College Stuff\\0. Programming stuff\\Python\\image_processing\\KTM.jpeg')
print(matrix)

img1 = Image.fromarray(matrix)
img1.show()

for i in range(len(matrix)):
    # print(matrix[i])
    for j in range (len(matrix[i])):
        matrix[i][j] = np.average(matrix[i][j])

img2 = Image.fromarray(matrix)

img2.show()
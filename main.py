import numpy as np
import matplotlib.pyplot as plt
# # define normalized 2D gaussian
# def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
#     return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

# size = 1

# x = np.linspace(-size, size, num=7)
# y = np.linspace(-size, size, num=7)
# x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
# z = gaus2d(x, y)
# mx =  max(z.flatten())
# mn =  min(z.flatten())
# z = (z-mn)/(mx-mn)
# print(z)
# plt.figure()
# plt.imshow(z)
# plt.show()

sigma = 2
tmp_size = 3

size = sigma * tmp_size + 1
#generate distribution range(0,1) and size
x = np.arange(0, size, 1, np.float32)
y = x[:, np.newaxis]
x0 = y0 = size // 2
#normaliz
g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

plt.figure()
plt.imshow(g)
plt.show()
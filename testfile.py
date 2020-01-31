import pandas as pd
import numpy as np

data1 = np.genfromtxt("./ima/20200115.dat",
                     dtype=None,
                     delimiter=' ',
                     usecols=(5, 7, 8, 9, 10, 11, 12))

data2 = np.genfromtxt("./ima/20200116.dat",
                       dtype=None,
                       delimiter=' ',
                       usecols=(5, 7, 8, 9, 10, 11, 12))

print(data1.shape)
print(data2.shape)
datac = np.concatenate((data1, data2), axis=0)
print(datac.shape)

# for row in data1:
#     np.concatenate((a, row), axis=0)

# a = np.array([[1,2],[2,3]])
# asplit = np.split(a,2)
# print(asplit)

# data2 = np.genfromtxt("./ima/20200121.csv",
#                      dtype=None,
#                      delimiter=' ',
#                      usecols=(5, 7, 8, 9, 10, 11, 12))
# print(data2)
# print(data2.shape)
#
# if data1.all() == data2.all():
#     print("wow")
#
# filename = "./ima/20200120.csv"
# print(filename[:-4])
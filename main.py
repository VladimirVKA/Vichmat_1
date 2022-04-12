import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

print('Задание 1')
matrix = np.random.randint(0,11,(7, 7),int)
print(matrix)
matrix_tr = np.transpose(matrix)
print(matrix_tr)
opr = np.linalg.det(matrix_tr)
print(int(opr))

print('Задание 2')
matrix_1 =np.random.randint(0,10,(2,3),int)
matrix_2 =np.random.randint(0,10,(3,2),int)
print(matrix_1)
print(matrix_2)
print(np.dot(matrix_1, matrix_2))

print('Задание 3')
M1 = np.array([[3, 2, 1], [3, 3, 2], [5, 5, 3]]) # Матрица (левая часть системы)
v1 = np.array([5, 7, 11]) # Вектор (правая часть системы)
res = np.linalg.solve(M1, v1)
print(round(res[0]),round(res[1]),round(res[2]))
print(np.dot(M1, res))

print('Задание 4')
res_4 = integrate.quad(lambda x:math.pow((1 + 2*math.pow(math.sin(x),2)),-1), 0, math.pi/4)
print(res_4)

print('Задание 5')
def f(y,x):
    x=y
    return y*y
def h(y):
    z = y+2
    return float(z)
v, err = integrate.dblquad(f, -1, 2, lambda y:y*y,lambda y:y+2)
print(v)

print('Задание 6')
X = np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)
C, L = 1-np.cos(X), np.sqrt(-3*X)

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="Cos Function")
plt.plot(X, L, color="red", linewidth=2.5, linestyle="-", label="Sqrt Function")
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$f(x)$', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig('figure_with_legend.png')
plt.show()
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

def MakeData(number, noise):
    if number == 0:
        X, y = ds.make_circles(n_samples=100, random_state= 34, noise=noise)
    elif number == 1:
        X, y = ds.make_moons(n_samples=100, random_state= 17, noise = noise)
    elif number == 2:
        X, y = ds.make_blobs(centers= 2, n_samples=100, random_state= 17, n_features=2)
    elif number == 3:
        rng = np.random.RandomState(0)
        X = rng.randn(200, 2)
        y = np.logical_xor(X[:, 0]>0, X[:, 1]>0)
    return X, y


def PolyFourthDeg(point, flag):
    x = point[0]
    y = point[1]
    poly = [1, x, y, x**2, y**2, x*y, x**2*y, x*y**2, x**2*y**2, x**3*y, x*y**3, x**3, y**3, x**4, y**4]
    if flag is True:
        poly = [-1*i for i in poly]
    return poly

A_set = []; B_set = []
scale = 3
size = 10
dataMaker = 3
X, y = MakeData(dataMaker, 0.05)
for i in range(len(y)):
    if y[i] == 1:
        A_set.append(X[i]*scale)
    else:
        B_set.append(X[i]*scale)
pointsNum_A = len(A_set)
pointsNum_B = len(B_set)
A_set = np.array(A_set)
B_set = np.array(B_set)
plt.scatter(A_set[:, 0], A_set[:, 1])
plt.scatter(B_set[:, 0], B_set[:, 1])
c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 1]
b_ub = [-1 for i in range(pointsNum_B + pointsNum_A)] \
       + [0]
#print(A_set)
A_ub = [PolyFourthDeg(i, False) + [-1] for i in A_set] + [PolyFourthDeg(j, True) + [-1] for j in B_set] \
        +[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]
solution = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = (None, None))
for key, val in solution.items():
    print(key, val)
    if key == 'x':
        q = [sum(i) for i in A_ub*val] #used res
        print('A*x', q)

solution = solution['x'].tolist()
delta = 0.01

xrange = np.arange(-size, size, delta)
yrange = np.arange(-size, size, delta)
X, Y = np.meshgrid(xrange,yrange)
# F is lhs, rhs is 0
F = solution[0]*1 + solution[1]*X+ solution[2]*Y+ solution[3]*X**2+ \
    solution[4]*Y**2+ solution[5]*X*Y+ solution[6]*X**2*Y+ solution[7]*X*Y**2+ solution[8]*X**2*Y**2\
    +solution[9]*X**3*Y+ solution[10]*X*Y**3+ solution[11]*X**3+ solution[12]*Y**3+ solution[13]*X**4+ solution[14]*Y**4
plt.contour(X, Y, F, [0])
plt.show()
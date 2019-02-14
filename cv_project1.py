import numpy as np
from sklearn.preprocessing import normalize
import math

# import txt files into matrices
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_1 = np.loadtxt('3Dpoints_part1.txt') # part 1
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2

# data normalization
# two_D_pts_normalized = normalize(two_D_pts, norm='l2')
# three_D_pts_1_normalized = normalize(three_D_pts_1, norm='l2')
# three_D_pts_2_normalized = normalize(three_D_pts_2, norm='l2')

# rename variables
m = two_D_pts
M_1 = three_D_pts_1
M_2 = three_D_pts_2

# split data for calibration (c) and error analysis (e) (approx. 80/20)
m_c = m[0:58]
m_e = m[58:73]
M_1_c = M_1[0:58]
M_1_e = M_1[58:73]
M_2_c = M_2[0:58]
M_2_e = M_2[58:73]
# print(M_1_c.shape)
# print(M_1_e.shape)

# form matrix A
N = len(m_c) # pairs of points
A = np.zeros((2*N, 12), dtype=np.float64)

for i in range(N):
	X,Y,Z = M_1_c[i] # 3D object points
	u,v = m_c[i] # 2D image points
	row_u = np.array([X,Y,Z,1,0,0,0,0,-X*u,-Y*u,-Z*u,-u])
	row_v = np.array([0,0,0,0,X,Y,Z,1,-X*v,-Y*v,-Z*v,-v])
	A[2*i] = row_u
	A[(2*i)+1] = row_v

# print(A.shape)
# print(A[0:10])

# SVD
u, s, vh = np.linalg.svd(A, full_matrices=True)

# find v (last column of V matrix times alpha scalar)
v_ = vh[:,11]
alpha = 1/math.sqrt(v_[8]**2 + v_[9]**2 + v_[10]**2)
print(alpha)
v = alpha * v_
print(v.shape)
result = A*v
print(result)

# use v to make P matrix

# project 3D to 2D using the projection matrix

# calculate error (m_pm vs m_e)
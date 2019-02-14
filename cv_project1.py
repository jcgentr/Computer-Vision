import numpy as np
import math
import cv2

# import txt files into matrices
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_1 = np.loadtxt('3Dpoints_part1.txt') # part 1
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2

# data normalization
# twoDsums = np.sum(two_D_pts, axis=0)
# x_avg = twoDsums[0] / len(two_D_pts)
# y_avg = twoDsums[1] / len(two_D_pts)
# d = 0
# for i in range(len(two_D_pts)):
# 	d += math.sqrt((two_D_pts[i][0] - x_avg)**2 + (two_D_pts[i][1] - y_avg)**2)
# d_avg = d / len(two_D_pts)
# print(x_avg,y_avg,d_avg)
# H_2D = np.zeros((3,3))
# row1 = np.array([math.sqrt(2)/d_avg, 0, -(math.sqrt(2)*x_avg)/d_avg])
# row2 = np.array([0, math.sqrt(2)/d_avg, -(math.sqrt(2)*y_avg)/d_avg])
# row3 = np.array([0, 0, 1])
# H_2D[0], H_2D[1], H_2D[2] = row1, row2, row3
# print(H_2D)
# print(two_D_pts[0])

# print(np.dot(H_2D, np.transpose(np.array([two_D_pts[0][0], two_D_pts[0][1], 1]))))
# twoDnorm = np.zeros((len(two_D_pts), 2))
# for i in range(len(two_D_pts)):
# 	 row = np.dot(H_2D, np.transpose(np.array([two_D_pts[i][0], two_D_pts[i][1], 1])))
# 	 row = row[0:2]
# 	 twoDnorm[i] = np.transpose(row)

twodnorm = np.zeros((len(two_D_pts),2))
cv2.normalize(two_D_pts,twodnorm)

# HOW DO I UN-NORMALIZE??
# print(cv2.normalize(twodnorm,None,np.max(two_D_pts),0,cv2.NORM_MINMAX)[0:3])
# print(two_D_pts[0:3])

# print("manual norm:\n", twoDnorm[0:3])
# print("cv norm:\n", twodnorm[0:3])

threednorm = np.zeros((len(three_D_pts_1),3))
cv2.normalize(three_D_pts_1,threednorm)

threednorm_2 = np.zeros((len(three_D_pts_2),3))
cv2.normalize(three_D_pts_2,threednorm_2)

def main(m, M_1, M_2):
	# split data for calibration (c) and error analysis (e) (approx. 80/20)
	m_c = m[0:58] # 58 pts
	m_e = m[58:73] # 14 pts
	M_1_c = M_1[0:58]
	M_1_e = M_1[58:73]
	M_2_c = M_2[0:58]
	M_2_e = M_2[58:73]

	# form matrix A
	N = len(m_c) # pairs of points
	A = np.zeros((2*N, 12), dtype=np.float64)

	for i in range(N):
		X,Y,Z = M_1_c[i] # 3D object point
		u,v = m_c[i] # 2D image point
		row_u = np.array([X,Y,Z,1,0,0,0,0,-X*u,-Y*u,-Z*u,-u])
		row_v = np.array([0,0,0,0,X,Y,Z,1,-X*v,-Y*v,-Z*v,-v])
		A[2*i] = row_u
		A[(2*i)+1] = row_v

	# SVD
	u, s, vh = np.linalg.svd(A, full_matrices=True)

	# find v (last column of V matrix times alpha scalar)
	v_ = vh[11,:]
	alpha = 1/math.sqrt(v_[8]**2 + v_[9]**2 + v_[10]**2)
	v = alpha * v_
	# print("alpha:", alpha)
	# print("v shape:", v.shape)
	# print("v:", v)
	result = np.dot(A,v)
	print("A dot v:", np.linalg.norm(result))

	# use v to make P matrix
	P = np.zeros((3,4), dtype=np.float64)
	row1 = np.array([v[0],v[1],v[2],v[3]])
	row2 = np.array([v[4],v[5],v[6],v[7]])
	row3 = np.array([v[8],v[9],v[10],v[11]])
	P[0], P[1], P[2] = row1, row2, row3
	print("Projection Matrix:\n", P)
	print("Shape of projection matrix:\n", P.shape)

	# project 3D to 2D using the projection matrix
	threeDpt = np.zeros((4,1), dtype=np.float64)
	sum_error = 0
	for i in range(len(m_e)):
		threeDpt[0] = M_1_e[i][0]
		threeDpt[1] = M_1_e[i][1]
		threeDpt[2] = M_1_e[i][2]
		threeDpt[3] = 1
		# projection
		twoDpt = np.dot(P,threeDpt)
		twoDpt = twoDpt/twoDpt[2]
		# print("Compare:\n", twoDpt[0:2], "\n", m_e[i])
		sum_error += np.linalg.norm([(twoDpt[0][0] - m_e[i][0]),(twoDpt[1][0] - m_e[i][1])])
	print("Total projection error:", sum_error)
	return P

# invoke main function with norm. and unnorm. data
P_norm = main(twodnorm, threednorm, threednorm_2)
P_reg = main(two_D_pts, three_D_pts_1, three_D_pts_2)


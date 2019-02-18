import numpy as np
import math
import cv2
from sklearn.utils import shuffle
from normalization import normalize

# import txt files into matrices and shuffle
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_1 = np.loadtxt('3Dpoints_part1.txt') # part 1
two_D_pts, three_D_pts_1 = shuffle(two_D_pts, three_D_pts_1, random_state=3)

# normalize data
twoDnorm, threeDnorm, H_2D, H_3D = normalize(two_D_pts, three_D_pts_1)

def main(m, M_1, isNormalized, H_2D=H_2D, H_3D=H_3D, two_D_pts=two_D_pts, three_D_pts_1=three_D_pts_1):
	# split data for calibration (c) and error analysis (e) (approx. 80/20)
	m_c = m[0:58] # 58 pts
	m_e = two_D_pts[58:73] # 14 pts
	M_1_c = M_1[0:58]
	M_1_e = three_D_pts_1[58:73]

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
	print("alpha:", alpha)
	v = alpha * v_
	result = np.matmul(A,v_)
	print("A matmul v:", np.linalg.norm(result))
	# print(result.shape)

	# use v to make P matrix
	P = np.zeros((3,4), dtype=np.float64)
	row1 = np.array([v[0],v[1],v[2],v[3]])
	row2 = np.array([v[4],v[5],v[6],v[7]])
	row3 = np.array([v[8],v[9],v[10],v[11]])
	P[0], P[1], P[2] = row1, row2, row3
	
	# if normalized, denormalize P matrix
	if isNormalized:
		P = np.matmul(np.matmul(np.linalg.inv(H_2D), P), H_3D)
		alpha = 1/math.sqrt(P[2][0]**2 + P[2][1]**2 + P[2][2]**2)
		P = alpha * P
	if P[2][3] > 0:
		P = -1 * P
	print("Projection Matrix:\n", P)

	# project 3D to 2D using the projection matrix
	threeDpt = np.zeros((4,1), dtype=np.float64)
	sum_error = 0
	for i in range(len(m_e)):
		threeDpt[0] = M_1_e[i][0]
		threeDpt[1] = M_1_e[i][1]
		threeDpt[2] = M_1_e[i][2]
		threeDpt[3] = 1
		# projection
		twoDpt = np.matmul(P,threeDpt)
		twoDpt = twoDpt/twoDpt[2]
		gt_2D = np.transpose(np.matrix(m_e[i]))
		# print("Compare:\n", twoDpt[0:2], "\n", gt_2D)
		diff = np.subtract(gt_2D, twoDpt[0:2])
		sum_error += np.linalg.norm(diff)
	print("Total projection error:", sum_error, "\n\n")
	return P


# invoke main function with norm. and orig. data
P_orig = main(two_D_pts, three_D_pts_1, False)
P_norm = main(twoDnorm, threeDnorm, True)


# find camera parameters
def camera_params(P):
	u0 = np.matmul([P[0][0],P[0][1],P[0][2]], [[P[2][0]],[P[2][1]],[P[2][2]]])
	v0 = np.matmul([P[1][0],P[1][1],P[1][2]], [[P[2][0]],[P[2][1]],[P[2][2]]])
	fu = math.sqrt(np.matmul([P[0][0],P[0][1],P[0][2]],[[P[0][0]],[P[0][1]],[P[0][2]]]) - u0**2)
	fv = math.sqrt(np.matmul([P[1][0],P[1][1],P[1][2]],[[P[1][0]],[P[1][1]],[P[1][2]]]) - v0**2)
	tx = (P[0][3] - u0*P[2][3]) / fu
	ty = (P[1][3] - v0*P[2][3]) / fv
	r1 = (np.array([[P[0][0]],[P[0][1]],[P[0][2]]]) - u0 * np.array([[P[2][0]],[P[2][1]],[P[2][2]]])) / fu
	r2 = (np.array([[P[1][0]],[P[1][1]],[P[1][2]]]) - v0 * np.array([[P[2][0]],[P[2][1]],[P[2][2]]])) / fv
	r3 = np.array([[P[2][0]],[P[2][1]],[P[2][2]]])

	print("u0:\n",u0[0])
	print("v0:\n",v0[0])
	print("fu:\n",fu)
	print("fv:\n",fv)
	print("tx:\n",tx[0])
	print("ty:\n",ty[0])
	print("r1:\n",r1)
	print("r2:\n",r2)
	print("r3:\n",r3)
	print("\n\n")


camera_params(P_orig)
camera_params(P_norm)



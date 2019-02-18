import numpy as np
import math
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from normalization import normalize

# import txt files into matrices and shuffle
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2
two_D_pts, three_D_pts_2 = shuffle(two_D_pts, three_D_pts_2, random_state=3)

# Step 1: randomly partition
def partition(rs, two_D_pts=two_D_pts, three_D_pts_2=three_D_pts_2):
	K = 7
	N = len(two_D_pts) # 72 pairs of pts
	test_portion = (N - K) / N
	train_2D, test_2D, train_3D, test_3D = train_test_split(two_D_pts, three_D_pts_2, random_state=rs, test_size=test_portion)
	return train_2D, test_2D, train_3D, test_3D

# Step 2: compute the projection matrix P using the training set
def main(train_2D, test_2D, train_3D, test_3D, inliers_max, H_2D, H_3D):
	m_c = train_2D
	m_e = test_2D
	M_1_c = train_3D
	M_1_e = test_3D

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
	try:
		alpha = 1/math.sqrt(v_[8]**2 + v_[9]**2 + v_[10]**2)
	except ZeroDivisionError:
		alpha = 1/math.sqrt((v_[8]**2 + v_[9]**2 + v_[10]**2) + 0.5)
	print("alpha:", alpha)
	v = alpha * v_
	# print("A matmul v:", np.linalg.norm(result))

	# print(result.shape)

	# use v to make P matrix
	P = np.zeros((3,4), dtype=np.float64)
	row1 = np.array([v[0],v[1],v[2],v[3]])
	row2 = np.array([v[4],v[5],v[6],v[7]])
	row3 = np.array([v[8],v[9],v[10],v[11]])
	P[0], P[1], P[2] = row1, row2, row3
	
	# denormalize P matrix
	P = np.matmul(np.matmul(np.linalg.inv(H_2D), P), H_3D)
	try:
		alpha = 1/math.sqrt(P[2][0]**2 + P[2][1]**2 + P[2][2]**2)
	except ZeroDivisionError:
		alpha = 2
	P = alpha * P
	if P[2][3] > 0:
		P = -1 * P

	print("Projection Matrix:\n", P)

	# project 3D to 2D using the projection matrix
	threeDpt = np.zeros((4,1), dtype=np.float64)
	sum_error = 0
	# Step 3: for each point in the testing set, compute its projection error
	threshold = 3
	inliers_count = 0
	two_D_inliers = []
	three_D_inliers = []
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
		proj_err = np.linalg.norm(diff)
		# print("Projection error:", proj_err)
		if proj_err < threshold:
			two_D_inliers.append(m_e[i])
			three_D_inliers.append(M_1_e[i])
			inliers_count += 1
		sum_error += proj_err

	print("Total projection error:", sum_error, "\n\n")
	
	if inliers_count > inliers_max:
		inliers_max = inliers_count
	return P, inliers_max, two_D_inliers, three_D_inliers, sum_error


inliers_max = 0
two_D_inliers = []
three_D_inliers = []
pe = 100000
best_rs = 0
for i in range(200):
	
	rs = np.random.randint(0, 100000)
	train_2D, test_2D, train_3D, test_3D = partition(rs)

	# normalize training data
	train_2D, train_3D, H_2D, H_3D = normalize(train_2D, train_3D)

	P, ic, twoDi, threeDi, PE = main(train_2D, test_2D, train_3D, test_3D, inliers_max, H_2D, H_3D)
	
	if ic > inliers_max:
		inliers_max = ic
		best_rs = rs
		pe = PE

	# capture inliers
	for j in range(len(twoDi)):
		two_D_inliers.append(twoDi[j])
		three_D_inliers.append(threeDi[j])

print("Max inliers:", inliers_max)
print("Lowest Projection Error:", pe)

if inliers_max > 0:
	two_D_inliers = np.unique(two_D_inliers, axis=0)
	three_D_inliers = np.unique(three_D_inliers, axis=0)

train_2D, test_2D, train_3D, test_3D = partition(best_rs)

# train_2D = np.vstack([train_2D, two_D_inliers])
# train_3D = np.vstack([train_3D, three_D_inliers])
# train_2D = np.unique(train_2D, axis=0)
# train_3D = np.unique(train_3D, axis=0)
# test_2D = np.vstack([test_2D, two_D_inliers])
# test_3D = np.vstack([test_3D, three_D_inliers])

# normalize training data
train_2D, train_3D, H_2D, H_3D = normalize(train_2D, train_3D)

print(len(train_2D))
print(len(train_3D))
print(len(test_2D))
print(len(test_3D))
print(len(two_D_inliers))
print(len(three_D_inliers))
# print(two_D_inliers[0:5])
# print(three_D_inliers)

P, ic, twoDi, threeDi, PE = main(train_2D, test_2D, train_3D, test_3D, inliers_max, H_2D, H_3D)

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


camera_params(P)



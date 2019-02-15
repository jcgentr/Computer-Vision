import numpy as np
import math
import cv2
from sklearn.utils import shuffle

# import txt files into matrices and shuffle
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2
two_D_pts, three_D_pts_2 = shuffle(two_D_pts, three_D_pts_2, random_state=3)

# data normalization
# for 2D pts
twoDsums = np.sum(two_D_pts, axis=0)
x_avg = twoDsums[0] / len(two_D_pts)
y_avg = twoDsums[1] / len(two_D_pts)
d = 0
for i in range(len(two_D_pts)):
	d += math.sqrt((two_D_pts[i][0] - x_avg)**2 + (two_D_pts[i][1] - y_avg)**2)
d_avg = d / len(two_D_pts)

H_2D = np.zeros((3,3))
row1 = np.array([math.sqrt(2)/d_avg, 0, -(math.sqrt(2)*x_avg)/d_avg])
row2 = np.array([0, math.sqrt(2)/d_avg, -(math.sqrt(2)*y_avg)/d_avg])
row3 = np.array([0, 0, 1])
H_2D[0], H_2D[1], H_2D[2] = row1, row2, row3

twoDnorm = np.zeros((len(two_D_pts), 2))
for i in range(len(two_D_pts)):
	 row = np.matmul(H_2D, np.transpose(np.array([two_D_pts[i][0], two_D_pts[i][1], 1])))
	 row = row[0:2]
	 twoDnorm[i] = np.transpose(row)

# for 3D pts
threeDsums = np.sum(three_D_pts_2, axis=0)
X_avg = threeDsums[0] / len(three_D_pts_2)
Y_avg = threeDsums[1] / len(three_D_pts_2)
Z_avg = threeDsums[2] / len(three_D_pts_2)
D = 0
for i in range(len(three_D_pts_2)):
	D += math.sqrt((three_D_pts_2[i][0] - X_avg)**2 + (three_D_pts_2[i][1] - Y_avg)**2 + (three_D_pts_2[i][2] - Z_avg)**2)
D_avg = D / len(three_D_pts_2)

H_3D = np.zeros((4,4))
row1 = np.array([math.sqrt(3)/D_avg, 0, 0, -(math.sqrt(3)*X_avg)/D_avg])
row2 = np.array([0, math.sqrt(3)/D_avg, 0, -(math.sqrt(3)*Y_avg)/D_avg])
row3 = np.array([0, 0, math.sqrt(3)/D_avg, -(math.sqrt(3)*Z_avg)/D_avg])
row4 = np.array([0, 0, 0, 1])
H_3D[0], H_3D[1], H_3D[2], H_3D[3] = row1, row2, row3, row4

threeDnorm = np.zeros((len(three_D_pts_2), 3))
for i in range(len(three_D_pts_2)):
	 row = np.matmul(H_3D, np.array([[three_D_pts_2[i][0]], [three_D_pts_2[i][1]], [three_D_pts_2[i][2]], [1]]))
	 row = row[0:3]
	 threeDnorm[i] = np.transpose(row)

print(two_D_pts[0:10])
print(twoDnorm[0:10])
print(three_D_pts_2[0:10])
print(threeDnorm[0:10])
# M_2_c = M_2[0:58]
# M_2_e = M_2[58:73]
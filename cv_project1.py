import numpy as np
from sklearn.preprocessing import normalize

# import txt files into matrices
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_1 = np.loadtxt('3Dpoints_part1.txt') # part 1
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2

# data normalization
two_D_pts_normalized = normalize(two_D_pts, norm='l2')
three_D_pts_1_normalized = normalize(three_D_pts_1, norm='l2')
three_D_pts_2_normalized = normalize(three_D_pts_2, norm='l2')

# rename variables
m = two_D_pts_normalized
M_1 = three_D_pts_1_normalized
M_2 = three_D_pts_2_normalized

# split data for calibration (c) and error analysis (e) (approx. 80/20)
m_c = m[0:58]
m_e = m[58:73]
M_1_c = M_1[0:58]
M_1_e = M_1[58:73]
M_2_c = M_2[0:58]
M_2_e = M_2[58:73]
print(M_1_c.shape)
print(M_1_e.shape)

# SVD
# u, s, vh = np.linalg.svd(a, full_matrices=True)

# find projection matrix v'

# project 3D to 2D using the projection matrix

# calculate error (m_pm vs m_e)
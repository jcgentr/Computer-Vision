import numpy as np
import math
import cv2
from sklearn.utils import shuffle
from normalization import normalize

# import txt files into matrices and shuffle
two_D_pts = np.loadtxt('2Dpoints.txt')
three_D_pts_2 = np.loadtxt('3Dpoints_part2.txt') # part 2
two_D_pts, three_D_pts_2 = shuffle(two_D_pts, three_D_pts_2, random_state=3)

# normalize data
twoDnorm, threeDnorm, H_2D, H_3D = normalize(two_D_pts, three_D_pts_2)

# M_2_c = M_2[0:58]
# M_2_e = M_2[58:73]
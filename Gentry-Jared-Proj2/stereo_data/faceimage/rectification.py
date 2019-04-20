import numpy
import cv2

def rectification(T,Wr,Wl,R,imgL,imgR):
 
    ##Step 1 ##############
    ##construct Rlrect
    v1 = T/numpy.linalg.norm(T)
    v1 = numpy.transpose(v1)

    temp = numpy.array([[0,0,1]])
    v2 = numpy.cross(v1,temp)

    v3 = numpy.cross(v1,v2)

    Rlrect = numpy.concatenate((v1,v2,v3), axis=0)
    Rlrect = -1 * Rlrect
    # print(v1,v1.shape)
    # print(v2,v2.shape)
    # print(v3,v3.shape)
    # print(Rlrect, Rlrect.shape)

    ##Step 2 ##########
    ##image reprojection left
    Wrect = (Wr + Wl) / 2
    
    # downsample to avoid holes
    Wrect /= 1.1
    # Wrect[0][2] += 2
    # Wrect[1][2] += 2

    row,col,ch = imgL.shape
    
    newImgL=numpy.zeros((row+100,col+100,3), numpy.uint8) # adding padding here if needed

    H_l = numpy.matmul(Wrect,Rlrect)
    H_l = numpy.matmul(H_l,numpy.linalg.inv(Wl)) 
    # print(H_l, H_l.shape)
    
    #calculating the position of the new points for the Left image
    min_x = 100 
    min_y = 100
    offset = numpy.array([[-83],[-2],[1]])
    newPointsL=[]
    for i in range(0,row):
        for j in range(0,col):
            temp = numpy.array([[i],[j],[1]])
            temp = numpy.matmul(H_l,temp)
            temp = temp + offset # make sure picture can be seen
            temp = temp.astype(int)
            x = temp[0]
            y = temp[1]
            if x < min_x:
            	min_x = x
            if y < min_y:
            	min_y = y
            newPointsL.append(temp)
            # print(x,y)
            newImgL[x+13,y] = imgL[i][j] # adding 13 to x moves image down

    print("offset left:", min_x,min_y)
    # print(newPointsL[0:10], len(newPointsL))
    # cv2.imshow('Original Left Image', imgL)
    # cv2.imshow('New Left Image',newImgL)
    cv2.imwrite('oldleft.png',imgL)
    cv2.imwrite('newleft.png',newImgL)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ##Step 3 #####
    #calculate Rrrect
#    Rrrect = numpy.matmul(Rlrect,R)
#    # print(Rrrect, Rrrect.shape)
#
#    row,col,ch = imgR.shape
##    print(imgR.shape)
##    print(imgR[0][1])
#    newImgR=numpy.zeros((row,col,3), numpy.uint8)
#
#    H_r = numpy.matmul(Wrect,Rrrect)
#    H_r = numpy.matmul(H_r,numpy.linalg.inv(Wr))
#    # print(H_r, H_r.shape)
#
#    #calculating the position of the new points for the Left image
#    min_x = 100
#    min_y = 100
#    offset = numpy.array([[76],[75],[1]])
#    newPointsR=[]
#    for i in range(0,row):
#        for j in range(0,col):
#            temp = numpy.array([[i],[j],[1]])
#            temp = numpy.matmul(H_r,temp)
#            temp = temp + offset
#            temp = temp.astype(int)
#            x = temp[0]
#            y = temp[1]
#            if x < min_x:
#                min_x = x
#            if y < min_y:
#                min_y = y
#            newPointsR.append(temp)
#            # print(x,y)
#            newImgR[x,y] = imgR[i][j]
#
#    print("offset right:", min_x,min_y)
#    # print(newPointsR[0:10], len(newPointsR))
#    # cv2.imshow('Original Right Image', imgR)
#    # cv2.imshow('New Right Image',newImgR)
#    cv2.imwrite('oldright.png',imgR)
#    cv2.imwrite('newright.png',newImgR)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

##testing function####################################################
r_left = numpy.array([[0.7552643,-0.65459403,0.03290133],
                      [0.13783671,0.09968584,-0.98542568],
                      [-0.64178723,-0.74880731,-0.16551967]])

r_right = numpy.array([[0.57126946,-0.8203492,0.02604603],
                       [0.16564024,0.08176389,-0.98279091],
                       [-0.80410364,-0.56575378,-0.1825924 ]])

R = numpy.matmul(r_left,numpy.transpose(r_right))

t_left = numpy.array([[5.963266213979592],[21.688085517596715],[-74.4625329655395]]) 

t_right = numpy.array([[12.806194509975613],[23.191573457787968],[-77.59926135564851]])

T = t_left - numpy.matmul(R,t_right)
T = -1 * T
# print(R)
# print(T)

Wl = numpy.array([[401.7720860407226, 0, 184.15045638684873], [0, 396.12224129404564, 118.14563010963332], [0, 0, 1]])
Wr = numpy.array([[412.2085911472977, 0, 187.92643715729065], [0, 407.7880359726471, 131.29161875061976], [0, 0 ,1]])

pointsL = cv2.imread('mark_leftface.bmp')
pointsR = cv2.imread('mark_rightface.bmp')

rectification(T,Wr,Wl,R,pointsL,pointsR)

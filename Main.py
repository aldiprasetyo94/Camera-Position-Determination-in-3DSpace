import cv2
import numpy as np
import cmath
import time

print('\n--- National Taiwan University of Science and Technology ---')
print('Name: 林超良 \t\t\t Department: Mechanical Engineering')
print('Student ID: M10803001 \t Assignment: Hw3 Determine Camera Position Relative to World Coordinate')
print('------------------------------------------------------------------------------------------------')



img1 = cv2.imread('ChosenSquare.jpg')
# cv2.cvtColor(img1,img1,cv2.COLOR_BGR2GRAY)


#DEFINE THE COORDINATE OF CHECKBOARD IN REAL
A_real = np.array([
    [0.,0.],
    [0.,30.],
    [30.,30.],
    [30.,0.]],np.float32)

B_real = np.array([
    [10.,10.],
    [10.,40.],
    [40.,40.],
    [40.,10.]],np.float32)

C_real = np.array([
    [10.,10.],
    [30.,10.],
    [30.,40.],
    [10.,40.]],np.float32)

#DEFINE THE COORDINATE OF CHECKBOARD IN PIXEL
A_img = np.array([
    [217,135],
    [248,311],
    [370,263],
    [365,117]],np.float32)

B_img = np.array([
    [440,117],
    [435,262],
    [561,307],
    [594,133]],np.float32)

C_img = np.array([
    [404,316],
    [288,383],
    [414,475],
    [524,375]],np.float32)

#For Projection P
#3DPoints
X1 = [50,0,0,1]
X2 = [50,50,0,1]
X3 = [0,50,0,1]
X4 = [0,50,50,1]
X5 = [0,0,50,1]
X6 = [50,0,50,1]
X7 = [30,0,30,1]

#2DPoints
uv1 = [205,382,1]
uv2 = [422,577,1]
uv3 = [605,374,1]
uv4 = [685,61,1]
uv5 = [402,58,1]
uv6 = [127,60,1]
uv7 = [280,190,1]

#Compute Homography for each side
h1, status1 = cv2.findHomography(A_real, A_img)
h2, status2 = cv2.findHomography(B_real, B_img)
h3, status3 = cv2.findHomography(C_real, C_img)
print('\nHomography of each square')
print('h1 (red square):')
print(h1)
print('\nh2 (blue square):')
print(h2)
print('\nh3 (yellow square):')
print(h3)


#Transpose Homography
h1 = np.transpose(h1)
h2 = np.transpose(h2)
h3 = np.transpose(h3)

#Create Matrix form for determining W (IAC)
def matrixA(h1,h2,h3):
    A=np.array([[(h1[0][0] * h1[1][0]), (h1[0][0] * h1[1][1] + h1[0][1] * h1[1][0]), (h1[0][0] * h1[1][2] + h1[0][2] * h1[1][0]), (h1[0][1] * h1[1][1]), (h1[0][1] * h1[1][2] + h1[0][2] * h1[1][1]), (h1[0][2] * h1[1][2])],
                [((h1[0][0] * h1[0][0]) - (h1[1][0] * h1[1][0])), 2 * ((h1[0][0] * h1[0][1]) - (h1[1][0] * h1[1][1])), 2 * ((h1[0][0] * h1[0][2]) - (h1[1][0] * h1[1][2])), ((h1[0][1] * h1[0][1]) - (h1[1][1] * h1[1][1])), 2 * ((h1[0][1] * h1[0][2]) - (h1[1][1] * h1[1][2])), ((h1[0][2] * h1[0][2]) - (h1[1][2] * h1[1][2]))]
                ,
                [(h2[0][0] * h2[1][0]), (h2[0][0] * h2[1][1] + h2[0][1] * h2[1][0]), (h2[0][0] * h2[1][2] + h2[0][2] * h2[1][0]), (h2[0][1] * h2[1][1]), (h2[0][1] * h2[1][2] + h2[0][2] * h2[1][1]), (h2[0][2] * h2[1][2])],
                [((h2[0][0] * h2[0][0]) - (h2[1][0] * h2[1][0])), 2 * ((h2[0][0] * h2[0][1]) - (h2[1][0] * h2[1][1])), 2 * ((h2[0][0] * h2[0][2]) - (h2[1][0] * h2[1][2])), ((h2[0][1] * h2[0][1]) - (h2[1][1] * h2[1][1])), 2 * ((h2[0][1] * h2[0][2]) - (h2[1][1] * h2[1][2])), ((h2[0][2] * h2[0][2]) - (h2[1][2] * h2[1][2]))]
                ,
                [(h3[0][0] * h3[1][0]), (h3[0][0] * h3[1][1] + h3[0][1] * h3[1][0]), (h3[0][0] * h3[1][2] + h3[0][2] * h3[1][0]), (h3[0][1] * h3[1][1]), (h3[0][1] * h3[1][2] + h3[0][2] * h3[1][1]), (h3[0][2] * h3[1][2])],
                [((h3[0][0] * h3[0][0]) - (h3[1][0] * h3[1][0])), 2 * ((h3[0][0] * h3[0][1]) - (h3[1][0] * h3[1][1])), 2 * ((h3[0][0] * h3[0][2]) - (h3[1][0] * h3[1][2])), ((h3[0][1] * h3[0][1]) - (h3[1][1] * h3[1][1])), 2 * ((h3[0][1] * h3[0][2]) - (h3[1][1] * h3[1][2])), ((h3[0][2] * h3[0][2]) - (h3[1][2] * h3[1][2]))]
            ])
    return A

A = matrixA(h1,h2,h3)

#FIND W (IAC)
def USV(matrixA):
    U, s, VH = np.linalg.svd(matrixA,full_matrices=True)
    V = VH.T.conj()
    w = np.array([[V[0][5], V[1][5], V[2][5]],
                  [V[1][5], V[3][5], V[4][5]],
                  [V[2][5], V[4][5], V[5][5]]])
    return w

w = USV(A)


def inverseW(w):
    # inverse of w
    W_inv = np.linalg.inv(w)
    # Normalize of w
    W_inv = W_inv / W_inv[2][2]
    return W_inv

inw = inverseW(w)


#Find K
#find a b c d e
def K(W_inv):
    c = W_inv[0,2]
    e = W_inv[1,2]
    d = cmath.sqrt((W_inv[1,1])-(e*e))
    b = ((W_inv[0,1]-(c*e))/d)
    a = np.sqrt(W_inv[0][0]-(b*b)-(c*c))

    # Get K
    K = np.array([[a, b, c],
                  [0, d, e],
                  [0, 0, 1]])
    return K

K = K(inw)
print('\n\nIntrinsic Parameter (K):')
print(K)




#PUV Matrix
zero = [0,0,0,0]
def uX(uv,noOfCol,X):
    result=[]
    for n in range(0,4):
        result.append(-1 * uv[noOfCol] * X[n])
    return result


#   | X1  0   -u1X1 |   Need at least 6 points
#   | 0   X1  -v1X1 |

Puvmatrix = [X1 + zero + uX(uv1, 0, X1),
             zero + X1 + uX(uv1, 1, X1),
             X2 + zero + uX(uv2, 0, X2),
             zero + X2 + uX(uv2, 1, X2),
             X3 + zero + uX(uv3, 0, X3),
             zero + X3 + uX(uv3, 1, X3),
             X4 + zero + uX(uv4, 0, X4),
             zero + X4 + uX(uv4, 1, X4),
             X5 + zero + uX(uv5, 0, X5),
             zero + X5 + uX(uv5, 1, X5),
             X6 + zero + uX(uv6, 0, X6),
             zero + X6 + uX(uv6, 1, X6),
             X7 + zero + uX(uv7, 0, X7),
             zero + X7 + uX(uv7, 1, X7)
             ]

# x = P . X
#P = Projective Camera
def Projection(PUVmatrix):
    U, s, VH = np.linalg.svd(PUVmatrix, full_matrices=True)
    V = VH.T.conj()

    P = np.array([[V[0][11], V[1][11], V[2][11], V[3][11]],
         [V[4][11], V[5][11], V[6][11], V[7][11]],
         [V[8][11], V[9][11], V[10][11], V[11][11]]
        ])
    P = P / P[2][3]

    return P

P =Projection(Puvmatrix)
print('P',P)


# Find RT --> RT = Kinv x P
RT = np.dot((np.linalg.inv(K)),P)

Length = np.sqrt((RT[0][0]*RT[0][0])+(RT[1][0]*RT[1][0])+(RT[2][0]*RT[2][0]))

RT[:]=[x/Length for x in RT]

newRT = np.array([[RT[0][0],RT[0][1],RT[0][2],RT[0][3]],
         [RT[1][0],RT[1][1],RT[1][2],RT[1][3]],
         [RT[2][0],RT[2][1],RT[2][2],RT[2][3]],
         [0,0,0,1]])

newRT_inv = np.linalg.inv(newRT)


#Camera Position
camX = newRT_inv[0][3]
camY = newRT_inv[1][3]
camZ = newRT_inv[2][3]

print('\nCamera Position:',camX,camY,camZ)


#Show the 3 selected square on the checkboard
cv2.imshow('Selected Square',img1)
cv2.waitKey(0)

time.sleep(300)
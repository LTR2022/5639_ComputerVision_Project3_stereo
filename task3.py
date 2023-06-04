import numpy as np
import cv2
from scipy.ndimage import filters
from scipy import ndimage
import matplotlib.pyplot as plt
import random
from scipy import linalg
import sys
import math

##########################
NCC_window_size = 8
epipole_number = 100
##########################

def harris_corner_detector(image, threshold, min_distance):
    sigma = 1.4
    # first compute derivatives
    img_x = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), img_x)
    img_y = np.zeros(image.shape)
    filters.gaussian_filter(image, (sigma, sigma), (1, 0), img_y)

    # compute Wxx, Wxy, Wyy
    Wxx = filters.gaussian_filter(img_x * img_x, sigma)
    Wxy = filters.gaussian_filter(img_x * img_y, sigma)
    Wyy = filters.gaussian_filter(img_y * img_y, sigma)

    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    harrisim = Wdet / Wtr

    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    coords = np.array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    index = np.argsort(candidate_values)
    allowed_locations = np.zeros(harrisim.shape)

    # non-maximum suppression
    allowed_locations[min_distance:-min_distance, min_distance:-min_distance] = 1
    results_coord = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            results_coord.append(coords[i])
            allowed_locations[(coords[i, 0] - min_distance):(coords[i, 0] + min_distance),
            (coords[i, 1] - min_distance):(coords[i, 1] + min_distance)] = 0
    return results_coord

def Calculate_NCC(window_left, window_right):
    N_L = sum(sum(window_left**2))**(1/2)
    N_R = sum(sum(window_right**2))**(1/2)
    ncc = sum(sum((window_left/N_L)*(window_right/N_R)))
    return ncc

def NCC_match_points(gray_left,gray_right,leftcorner,rightcorner,windowsize,threshold):
    m = len(leftcorner)
    n = len(rightcorner)
    gray_left = np.array(gray_left,dtype='uint32')
    gray_right = np.array(gray_right,dtype='uint32')
    NCC_value = np.zeros((m,n))
    for i in range (m):
        leftwindow = gray_left[(leftcorner[i][0] - windowsize):(leftcorner[i][0] + windowsize + 1), (leftcorner[i][1] - windowsize):(leftcorner[i][1] + windowsize + 1)]
        for j in range(n):
            rightwindow = gray_right[(rightcorner[j][0]-windowsize):(rightcorner[j][0]+windowsize+1), (rightcorner[j][1]-windowsize):(rightcorner[j][1]+windowsize+1)]
            NCC_value[i,j] = Calculate_NCC(leftwindow, rightwindow)
    ncc = NCC_value.tolist()
    matches = []
    for i in range (m*n):
        m_max = []
        for j in range(m):
            m_max.extend([max(ncc[j])])
        temp = max(m_max)
        if temp < threshold:
            break
        for a in range (m):
            for b in range(n):
                if ncc[a][b]==temp:
                    matches += [[a,b]]
                    ncc[a][b] = 0
                    a=m+1
                    b=n+1
                    break
    return matches

def draw_match_points(image_left, image_right, left_keypoints, right_keypoints, matches, linenumber):
    """
    Draws keypoints and their matches between two images.

    Args:
    - image_left (numpy array): The first image
    - image_right (numpy array): The second image
    - left_keypoints (list): Keypoint list of the left image
    - right_keypoints (list): Keypoint list of the right image
    - matches (list): List of matches between keypoints
    - linenumber (int): Number of matches to be drawn

    Returns:
    - None
    """
    # Create an empty image to hold the combined images
    width = image_left.shape[1] + image_right.shape[1]
    height = max(image_left.shape[0], image_right.shape[0])
    image = np.zeros([height, width, 3], dtype='uint8')

    # Copy the first image to the left side of the combined image
    for i in range(0, width):
        for j in range(0, height):
            if i < ((image.shape[1] / 2) - 1):
                image[j][i] = image_left[j][i]
                x = i
            else:
                image[j][i] = image_right[j][i - x - 2]

    # Draw the matches on the combined image
    plt.ion()
    plt.imshow(image, cmap='gray')
    for a in range(0, linenumber):
        temp_l = matches[a][0]
        temp_r = matches[a][1]
        plt.plot(left_keypoints[temp_l][1], left_keypoints[temp_l][0], '*')
        plt.plot(right_keypoints[temp_r][1] + image_left.shape[1], right_keypoints[temp_r][0], '*')
        plt.plot([left_keypoints[temp_l][1], right_keypoints[temp_r][1] + image_left.shape[1]],
                 [left_keypoints[temp_l][0], right_keypoints[temp_r][0]], linewidth=1)
    plt.ioff()
    plt.axis('off')
    plt.show()

    return 0

def rgbtohsv(img):
    m, n, k = img.shape
    r, g, b = cv2.split(img)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = np.zeros((m, n), np.float32)
    S = np.zeros((m, n), np.float32)
    V = np.zeros((m, n), np.float32)
    HSV = np.zeros((m, n, 3), np.float32)
    for i in range(0, m):
        for j in range(0, n):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            V[i, j] = mx
            if V[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = (V[i, j] - mn) / V[i, j]
            if mx == mn:
                H[i, j] = 0
            elif V[i, j] == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn)) + 360
            elif V[i, j] == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (V[i, j] - mn) + 120
            elif V[i, j] == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (V[i, j] - mn) + 240
            H[i, j] = H[i, j] / 2
    HSV[:, :, 0] = H[:, :]
    HSV[:, :, 1] = S[:, :]
    HSV[:, :, 2] = V[:, :]
    return HSV

def draw_inliers(image1, image2, cor_l, cor_r, ncc_match, F):
    threshold = 1
    inliers = []
    draw_p = min(len(ncc_match), 40)

    for c in range(draw_p):
        p_l = np.hstack([cor_l[ncc_match[c][0]], 1])
        p_r = np.hstack([cor_r[ncc_match[c][1]], 1])

        if np.dot(np.dot(p_l, F), p_r) < threshold:
            inliers.append(ncc_match[c])

    print(len(inliers))
    draw_match_points(image1, image2, cor_l, cor_r, inliers, len(inliers))
    return 0

def findFmatrix(x1,x2,n):
    x1=np.array(x1)
    x2=np.array(x2)
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[i, 1] * x2[i, 1], x1[i, 1] * x2[i, 0], x1[i, 1],
                x1[i, 0] * x2[i, 1], x1[i, 0] * x2[i, 0], x1[i, 0],
                x2[i, 1], x2[i, 0], 1]
    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # constrain F
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F / F[2, 2]

def Ransac_f_matrix(cor_l,cor_r,ncc_match):
    ransactimes = 1000
    ACC_result=sys.maxsize
    F_result=None
    ransanc_num=50
    for a in range(ransactimes):
        p1=[]
        p2=[]
        for b in range(ransanc_num):
          temp=random.randint(0, len(ncc_match)-1)
          p1.extend([cor_l[ncc_match[temp][0]]])
          p2.extend([cor_r[ncc_match[temp][1]]])
        F=findFmatrix(p1,p2,ransanc_num)
        temp1=0
        for c in range(len(ncc_match)):
            p_l=[]
            p_l.extend([cor_l[ncc_match[c][0]]])
            p_l=np.append(p_l,1)
            p_r = []
            p_r.extend([cor_r[ncc_match[c][1]]])
            p_r = np.append(p_r, 1)
            temp1+=abs(np.dot(np.dot(p_l.T,F),p_r))
        if temp1<ACC_result:
            ACC_result=temp1
            F_result=F
            #print([a,ACC_result])
    return F_result

def calculateepipole(point,F,m,n,linepoints):
    e_line=np.dot(F,point.T)
    # print(e_line)
    t = np.linspace(NCC_window_size, n-NCC_window_size-1, linepoints)
    # print(t)
    lt = np.array([(e_line[2] + e_line[0] * tt) / (-e_line[1]) for tt in t])
    # print(lt)
    ndx = (lt >= 0) & (lt < m)
    # print(ndx)
    t = np.reshape(t, (linepoints, 1))
    return t[ndx],lt[ndx]

def Compute_dense_map(leftimage,rightimage,F,fullimage):
    v=rgbtohsv(fullimage)[:,:,2]
    m, n= leftimage.shape
    horizontal_map=np.zeros((m,n))
    vertical_map=np.zeros((m,n))
    vector_map=np.zeros((m,n,3))
    leftimage = np.array(leftimage, dtype='uint32')
    rightimage = np.array(rightimage, dtype='uint32')
    temp_right=np.zeros((m+2*NCC_window_size,n))
    temp_right[NCC_window_size:-NCC_window_size,:]=rightimage
    max_h=0
    max_v=0
    max_sat=0
    # calculate epipole line
    for i in range(NCC_window_size,m-NCC_window_size-1):
    # for i in range(30,80):
        for j in range(NCC_window_size,n-NCC_window_size-1):
        # for j in range(30,80):
            line_x,line_y=calculateepipole(np.array([j,i,1]),F,m,n,epipole_number)
            # print(line_x, line_y)
            leftwindow = leftimage[(i-NCC_window_size):(i+NCC_window_size+1),(j-NCC_window_size):(j+NCC_window_size+1)]
            # search NCC on the line
            temp_ncc=0
            temp_match=[]
            for a in range(len(line_x)):
               pointx=int(line_x[a])
               pointy=int(line_y[a])
               for b in range(-3, 4):
                   rightwindow = temp_right[(pointy):(pointy+2*NCC_window_size+1),(pointx-NCC_window_size):(pointx+NCC_window_size+1)]
                   temp = Calculate_NCC(leftwindow,rightwindow)
                   if temp>temp_ncc and temp>0:
                          temp_ncc=temp
                          temp_match=[pointy,pointx]

            if temp_match!=[]:
                dif_x=abs(j-temp_match[1])
                dif_y=abs(i-temp_match[0])
                horizontal_map[i][j]=dif_x
                vertical_map[i][j]=dif_y
                hue=0
                if dif_x!=0:
                   hue=math.degrees(math.atan(dif_y/dif_x))
                if hue<0:
                    hue=360-abs(hue)
                sat=(dif_x**2+dif_y**2)**(1/2)
                vector_map[i][j][0] = hue
                vector_map[i][j][1] = sat
                if dif_x>max_h:
                   max_h=dif_x
                if dif_y>max_v:
                    max_v=dif_y
                if sat>max_sat:
                    max_sat=sat
        print(i)
    horizontal_map*=(255/max_h)
    vertical_map*=(255/max_v)
    #print(vertical_map[:0])
    for i in range(NCC_window_size,m-NCC_window_size-1):
        for j in range(NCC_window_size,n-NCC_window_size-1):
            if horizontal_map[i][j] == 0:
                temp=horizontal_map[i-1][j-1]+horizontal_map[i-1][j]+horizontal_map[i-1][j+1]+horizontal_map[i][j-1]+horizontal_map[i][j+1]+horizontal_map[i+1][j-1]+horizontal_map[i+1][j]+horizontal_map[i+1][j+1]
                temp/=8
                if temp>=10:
                    horizontal_map[i][j]=int(255-temp)
            else:
                horizontal_map[i][j]=int(255-horizontal_map[i][j])
            if vertical_map[i][j] == 0:
                temp=vertical_map[i-1][j-1]+vertical_map[i-1][j]+vertical_map[i-1][j+1]+vertical_map[i][j-1]+vertical_map[i][j+1]+vertical_map[i+1][j-1]+vertical_map[i+1][j]+vertical_map[i+1][j+1]
                temp/=8
                if temp>=10:
                    vertical_map[i][j]=int(255-temp)
            else:
                vertical_map[i][j]=int(255-vertical_map[i][j])
    vector_map[:,:,2]=v
    vector_map[:,:,1]*=(255/max_sat)

    # horizontal disparity component
    plt.ion()
    plt.imshow(horizontal_map,cmap ='gray')
    plt.ioff()
    plt.axis('off')
    plt.show()
    # vertical disparity component
    plt.ion()
    plt.imshow(vertical_map, cmap='gray')
    plt.ioff()
    plt.axis('off')
    plt.show()
    # disparity vector using color
    plt.ion()
    plt.imshow(vector_map, cmap='hsv')
    plt.ioff()
    plt.axis('off')
    plt.show()
    return 0


if __name__ == '__main__':
    # read images and convert it into gray map
    leftimage = cv2.imread("cast-left-1.jpeg")
    rightimage = cv2.imread("cast-right-1.jpeg")
    gray_left = cv2.cvtColor(leftimage, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rightimage, cv2.COLOR_BGR2GRAY)

    # Harris corner detection
    left_corners = harris_corner_detector(gray_left, 0.03, 8)
    right_corners = harris_corner_detector(gray_right, 0.03, 8)
    ncc_match_points = NCC_match_points(gray_left, gray_right, left_corners, right_corners, NCC_window_size, 0.99)

    print([len(left_corners), len(right_corners)])
    print(len(ncc_match_points))

    # visualize the all correspondenc results
    draw_match_points(gray_left, gray_right, left_corners, right_corners, ncc_match_points, len(ncc_match_points))

    # compute F matrix with RANSAC with 8 pts
    F = Ransac_f_matrix(left_corners, right_corners, ncc_match_points)
    print(f'Fundanmental Matrix is: {F}')
    draw_inliers(leftimage, rightimage, left_corners, right_corners, ncc_match_points, F)

    # compute dense disparity map and show the result
    Compute_dense_map(gray_left, gray_right, F, leftimage)
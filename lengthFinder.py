import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
#import matplotlib.animation as animation
#from matplotlib.animation import PillowWriter
import itertools
#from scipy import misc
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra
import math
import random as rng
from skimage.morphology import skeletonize
from skimage import data

#uses dijkstra's and convex hull to find the length of a root
def allpairsmaxminpath(imgPassed): 
    img2 = np.copy(imgPassed)
    img2 = np.uint8(img2)
    
    threshold = .5
    # Detect edges using Canny
    canny_output = cv2.Canny(img2, threshold, threshold * 2)
    
    plt.imshow(canny_output)
    # Find contours
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    #for i in range(len(contours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        #cv2.drawContours(drawing, contours, i, color)
        #cv2.drawContours(drawing, hull_list, i, color)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1]), dtype=np.uint8)
    
    #hull_list_copy = np.array(hull_list)
    
    #for j in range(0,hull_list_copy.shape[0]):
    #    for i in range(0,hull_list_copy[j].shape[0]):
    #        drawing[hull_list_copy[j][i][0][1]][hull_list_copy[j][i][0][0]] = 1
    #plt.imshow(drawing)
    ## 18 pixels turned on, but tuple is only (2, 13, 1, 2)
    #np.where(drawing == 1)[0].shape
    #hull_list_copy.shape
    hull_list_copy = []
    for i in range(0,len(hull_list)):
        for j in range(0,len(hull_list[i])):
            if not any(np.array_equal(hull_list[i][j], unique_arr) for unique_arr in hull_list_copy):
                hull_list_copy.append(hull_list[i][j])
            #hull_list_copy.append(hull_list[i][j])
    #np.array(hull_list_copy).shape
    hull_list_copy = np.array(hull_list_copy)
    hull_list_copy.shape
    
    img = np.uint8(np.copy(imgPassed))
    kernel = np.ones((5, 5),np.uint8)
    dilation = cv2.dilate(np.uint8(img),kernel,iterations = 1)
    #plt.imshow(dilation)
    for j in range(0,hull_list_copy.shape[0]):
        dilation[hull_list_copy[j][0][1]][hull_list_copy[j][0][0]] = 3
    #plt.imshow(dilation)
    
    
    
    img = np.copy(dilation)
    # A sparse adjacency matrix.
    # Two pixels are adjacent in the graph if both are painted.
    adjacency = dok_matrix((img.shape[0] * img.shape[1], img.shape[0] * img.shape[1]), dtype=bool)
            
    # The following lines fills the adjacency matrix by
    directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if not img[i, j]:
                continue
            
            for y_diff, x_diff in directions:
                if img[i + y_diff, j + x_diff]:
                    adjacency[to_index(i, j), to_index(i + y_diff, j + x_diff)] = True
            
    maxDist = 0
    maxPath = []
    #hull_list_copy[0][0].shape
    #hull_list_copy.shape
    
    ##so i think change this to a double for-loop
    for j in range(0,hull_list_copy.shape[0]):
        for k in range(j,hull_list_copy.shape[0]):
          
            #2 points we know are connected, hmmmmmmmmm
            source = to_index(hull_list_copy[j][0][1], hull_list_copy[j][0][0])
            target = to_index(hull_list_copy[k][0][1], hull_list_copy[k][0][0])
            
            #shortest path between the source and all other points in the image
            _, predecessors = dijkstra(adjacency, directed=False, indices=[source], unweighted=True, return_predecessors=True)
            
            predecessors[predecessors != -9999].shape
            
            #construct the path
            pixel_index = target
            pixels_path = []
            while pixel_index != source:
                pixels_path.append(pixel_index)
                pixel_index = predecessors[0, pixel_index]
                
                #for pixel_index in pixels_path:
                #    x, y = to_coordinates(pixel_index)
                #print(i, j)
                #    img[x, y] = 2
                
                #if this is now our longest shortest path, keep it
                if (len(pixels_path) > maxDist):
                    maxDist = len(pixels_path)
                    maxPath = np.copy(pixels_path)



    for pixel_index in maxPath:
        x, y = to_coordinates(pixel_index)
        #print(i, j)
        img2[x, y] = 2
    #plt.close()
    #plt.imshow(img2)
    #plt.show()
    
    return len(maxPath)


#isomorphism mapping to a single number
def to_index(y, x):
    return y * img.shape[1] + x


#isomorphism mapping back to coordinates
def to_coordinates(index):
    return math.floor(index / img.shape[1]), math.floor(index % img.shape[1])

#credit to https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python?rq=1
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    #cv2.imshow('labeled.png', labeled_img)
    #cv2.waitKey()
    plt.imshow(labeled_img)
    ##plt.imshow(labels)
    plt.show()


def imshow_one_component(labels, component, ims):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', labeled_img)
    
    imReduced = labeled_img
    # remove unwanted components
    imReduced[labels != component] = 0
    #imReduced = cv2.resize(imReduced, (200, 100), interpolation=cv2.INTER_CUBIC)
    #cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('image', 200, 100)
    #cv2.imshow('image', imReduced)
    #cv2.waitKey(15)
    #cv2.destroyAllWindows()
    #im = plt.imshow(imReduced)
    ims.append(imReduced)
    #plt.show()
    return ims


def getReducedImages(labels):
    ims = []
    ## labels[labels == 0] is just the background, 1st root is at labels==1
    for i in range(1, np.max(labels) + 1):
        component = i
        imReduced = np.copy(labels)
        imReduced[imReduced != component] = 0
        imReduced[imReduced != 0] = 1
        ims.append(imReduced)
        #plt.imshow(imReduced)
        #plt.show()
    return ims




# binary ground truth image
name = 'CLMB ground truth/Same Date, Different Resolutions/300 DPI/DOE.S300_T113_L1_19.05.17_134422_1_clmb_BW2.mat'
#name = 'CLMB ground truth/Same Date, Different Resolutions/300 DPI/DOE.S300_T113_L4_19.05.17_135352_1_clmb_BW2.mat'
#name = 'CLMB ground truth/Same Date, Different Resolutions/300 DPI/DOE.S300_T113_L5_19.05.17_135439_1_clmb_BW2.mat'

file = scipy.io.loadmat(name,  mat_dtype = True, squeeze_me = True, struct_as_record=False)
#>>> sorted(file.keys())
#['BW2', '__globals__', '__header__', '__version__']
img = file['BW2']

## erodes then dilates to remove noise, erases whole branches though...
kernel = np.ones((50, 50),np.uint8)
kernel = np.eye(50,dtype=np.uint8)[::-1]
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
#kernel = np.eye(5,dtype=np.uint8)[::-1]
#opening = cv2.morphologyEx(img, cv.MORPH_OPEN, kernel)
#plt.imshow(opening)
#dilation = cv2.dilate(img,kernel,iterations = 1)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
####plt.imshow(closing)
#plt.imshow(dilation)
#plt.show()

# >>> img.dtype
# dtype('<f8')
img = np.array(closing, dtype=np.uint8)
plt.imshow(img)
plt.show()

ret, labels = cv2.connectedComponents(img)

roots = []
roots = getReducedImages(labels)

lengths = []
for i in range(0, len(roots)):
    print("finding length of the", i, "th root\n")
    pathLen = allpairsmaxminpath(roots[i])
    print(pathLen, "\n")
    lengths.append(pathLen/300)

##print out lengths
for i in range(0, len(lengths)):
    print("root",i,"is of length", lengths[i],"\n")

#Diameter stuff::::
def getDiam(imgPassed):
    #plt.imshow(imgPassed)
    #plt.show()
    #
    #### divide out by max to descale label back to 1
    image = (np.copy(imgPassed)/np.max(imgPassed))
    #
    #### perform skeletonization
    skeleton = skeletonize(image)
    
    #### use dilation to get outline
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    dilated = cv2.dilate(image,kernel,iterations = 1)
    #plt.imshow(dilated)
    outline = dilated - image
    #plt.imshow(outline)
    
    skelAndOutline = np.copy(skeleton) * 2
    skelAndOutline[np.where(outline == 1)] = 1
    #plt.imshow(skelAndOutline)
    
    outlineYs = np.where(skelAndOutline == 1)[0]
    outlineXs = np.where(skelAndOutline == 1)[1]
    skelYs = np.where(skelAndOutline == 2)[0]
    skelXs = np.where(skelAndOutline == 2)[1]
    minDists = []
    k = 3
    for i in range(0, len(outlineYs)):
        minDist = float('inf')
        bestX = -1
        bestY = -1
        for j in range(0, len(skelYs)):
            euclideanDist = np.sqrt((outlineYs[i] - skelYs[j])**2 + (outlineXs[i] - skelXs[j])**2)
            ## radius is 1/2*diameter
            if(2*euclideanDist < minDist):
                minDist = 2*euclideanDist
                bestY = skelYs[j]
                bestX = skelXs[j]

        minDists.append(minDist)
        skelAndOutline[outlineYs[i]][outlineXs[i]] = k
        skelAndOutline[bestY][bestX] = k
        #plt.imshow(skelAndOutline)
        k = k+1
    #K = 5 ## the K largest values in our array 
    #highFive = np.sort(minDists)[-5:]
    #avgMaxedDiameters = np.average(highFive)
    
    skelAndOutline_hue = np.uint8(179*skelAndOutline/np.max(skelAndOutline))
    blank_ch = 255*np.ones_like(skelAndOutline_hue)
    skelAndOutline_img = cv2.merge([skelAndOutline_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    skelAndOutline_img = cv2.cvtColor(skelAndOutline_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    skelAndOutline_img[skelAndOutline==0] = 0
    
    plt.imshow(skelAndOutline_img)
    return np.average(minDists)/300

diams = []
for i in range(0, len(roots)):
    print("finding diameter of the", i, "th root\n")
    diam = getDiam(roots[i])
    print(diam, "\n")
    diams.append(diam)

##print out diams
for i in range(0, len(diams)):
    print("root",i,"is of length", diams[i],"\n")

#np.argmax(diams)
#plt.close()
#plt.imshow(roots[8])
#np.argmin(diams)
#plt.close()
#plt.imshow(roots[14])
#diams[14]
#getDiam(roots[8])
#getDiam(roots[14])
#plt.imshow(roots[1])
#getDiam(roots[1])
    
lengths = np.zeros(25)

##makes an image of roots each with a different color, and displays length and diameter 3 pixels up and 3 to the right
labelCopy = np.copy(labels).astype(float)
labelCopy[np.where(labels == 0)] = np.nan
plt.axis('off')
plt.imshow(labelCopy, cmap='jet')
for i in range(0, len(roots)):
    yx = (np.where(labels == i + 1))
    y = yx[0]
    x = yx[1]
    topy = np.min(y)
    topx = np.min(x[y == topy])
    txt = "Length: " + str(round(lengths[i], 2)) + " Diam: " + str(round(diams[i],2))
    plt.annotate(txt,xy=(topx,topy), xytext=(topx + 3,topy - 3), fontsize=8)

#np.max(labels)
    len(roots)
    len(lengths)
    len(diams)

plt.show()

#
#lengths = lengths[0:23]
#lengths = np.array(lengths)*300
#
#lengths[22]
#ims = []
#ims = imshow_one_component(labels,23, ims)
#len(ims)
#plt.imshow(ims[0])
#plt.imshow(roots[22])
#allpairsmaxminpath(roots[22])
#np.max(labels)
#len(roots)
#plt.close()


##my attempt to start directional closing based off the gradient of the image:
imgTest = np.float32(np.copy(labels))
sobelx = cv2.Sobel(imgTest,cv2.CV_64F,1,0) # Find x and y gradients
sobely = cv2.Sobel(imgTest,cv2.CV_64F,0,1)
mag = np.sqrt(sobelx**2.0 + sobely**2.0)
direction = np.arctan2(sobely, sobelx) * (180 / np.pi)
plt.imshow(mag)
plt.imshow(direction)
np.max(mag)

imgTest = (np.copy(img))
plt.imshow(imgTest)
kernel3 = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.uint8)
plt.imshow(imgTest[200:500, 250:750])
closing = cv2.morphologyEx(imgTest[200:500, 250:750], cv2.MORPH_CLOSE, kernel3)
plt.imshow(closing)
kernel3 = np.eye(15,dtype=np.uint8)[::-1]   ##the only way I can see to rotate the eye matrix to close these roots correctly in CLMB ground truth/Same Date, Different Resolutions/300 DPI/DOE.S300_T113_L1_19.05.17_134422_1_clmb_BW2.mat 
closing = cv2.morphologyEx(imgTest[200:500, 250:750], cv2.MORPH_CLOSE, kernel3)
plt.imshow(closing)
imgTest[200:500, 250:750] = closing
plt.imshow(imgTest)
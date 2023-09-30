import cv2 
import os 
import matplotlib.pyplot as plt
import numpy as np
import shapely
from rembg import remove

#========================FUNCTIONS==========================
def proccessIntersectionPoint(interPoints, x1, y1):
    '''
    interPoints: shapely intersection geometry, 
    x1: x-coord of point on baseline, 
    y1: y-coord of point on baseline, \n
    returns: LIST of all intersection points
             Shape [(x1,y1), (x2,y2),....]
    '''
    pointType = shapely.get_type_id(interPoints)
    allPoints = []
    if(pointType == 0):
        return [(interPoints.x, interPoints.y)]
    elif(pointType == 4):
        interPoints = interPoints.geoms
        for pt in interPoints:
            allPoints.append((pt.x, pt.y))
    elif(pointType == 1):
        all_x, all_y = interPoints.xy
        allPoints = list(zip(all_x, all_y))
        
    elif(pointType == 7): #Geometry collection
        allObjects = interPoints.geoms
        
        for obj in allObjects:
            coords = proccessIntersectionPoint(obj, x1, y1)
            
            allPoints.extend(coords)
    
    return allPoints


def getREMBGMask(image):
    return remove(image, post_process_mask=True)


#==========================MAIN============================

imageDir = "./" #place your image directory here

acceptedFileTypes = ["png", "tif", "jpeg"]


for image_name in os.listdir(imageDir):
    #check extension to proceed with processing
    if image_name.split(".")[-1].lower() in acceptedFileTypes:    
        print("Processing: ", image_name)
        
        #load image
        image = cv2.imread(imageDir + "/" + image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #create a horizontal line (TO-DO: PULL OUT TO A GENERAL FUNCTION)
        x = np.arange(0, image.shape[1], 1)
        y = np.full((image.shape[1]), fill_value = 300)
        
        # create blank image, black background
        blackBoard = np.full(gray.shape, fill_value=0, dtype="uint8")
        cv2.line(blackBoard, (0,200), (image.shape[1],200), (255), 1)
        
        #find points of intersection 
        interPoints = cv2.bitwise_and(gray, blackBoard)
        
        #locations of intersections
        inter_x, inter_y = np.where(interPoints != 0)
        
        coordinates = list(zip(inter_x, inter_y))
        
        print(coordinates)
        
        #Preview of intersection points
        plt.title("Intersection Points")
        plt.imshow(blackBoard | gray)
        plt.plot(inter_y, inter_x, "r.") #order different because of image axis orientation
        plt.show()
        
        #further processing....
    
        
#=========================TESTING=========================
# #find contours, return ALL contours in a list
# #structure is list[ list[ coordinate points]]
# contours, hier = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# #convert each contour to shapely Polygon
# contours = map(np.squeeze, contours)  # removing redundant dimensions
# polygons = map(shapely.geometry.Polygon, contours)  # converting to Polygons
# multipolygon = shapely.geometry.MultiPolygon(polygons)  # putting it all together in a MultiPolygon
     

# #create a horizontal line (TO-DO: PULL OUT TO A GENERAL FUNCTION)
# x = np.arange(0, image.shape[1], 1)
# y = np.full((image.shape[1]), fill_value = 300)

# #show the lines
# plt.imshow(image)
# plt.plot(x,y, "g-")
# plt.show()


# #convert line to shapely linstring
# stack = list(zip(x,y))
# lineString = shapely.geometry.LineString(stack)


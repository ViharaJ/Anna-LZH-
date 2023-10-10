'''
Failed to segment properly
'''
import cv2 
import os 
import numpy as np 
from rembg import remove, new_session
import pandas as pd
import matplotlib.pyplot as plt
       

def getRemBGMask(image, post_process=True):
    '''
    post_process default is True
    return  mask
    '''
    full_mask = remove(image, post_process_mask=post_process, only_mask=True)  
    
    p1 = np.full((image.shape[0]+2, image.shape[1]+2), fill_value=0, dtype="uint8")
    
    cv2.floodFill(full_mask, p1, (0,0), 0)
    
    p1 = cv2.bitwise_not(p1*255)
    
    plt.imshow(p1, cmap="gray")
    plt.show()
    return p1[1:-1, 1:-1]    
    

def postProcess(image):
    '''
    remove background with processing on mask
    return image with background removed (alpha channel included)
    '''
    return remove(image, post_process_mask=True, session=new_session("u2net"))    



#load image
image = cv2.imread("./Sample.jpeg", cv2.IMREAD_GRAYSCALE)

# background removed image 
rem_bg = postProcess(image)

# plot background removed image
plt.imshow(rem_bg, cmap ="gray")
plt.show()

# binarize
ret, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# get mask, then plot
mask = getRemBGMask(otsu)
plt.imshow(mask, cmap ="gray")
plt.show()



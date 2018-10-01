# Imports
import numpy as np
import cv2
import math

# Read the original image
img = cv2.imread('C:\\Users\\ASUS\\Pictures\\ngon5_4.jpg',1)

# Copy image  
crop_image = img

# Apply Gaussian blur
blur = cv2.GaussianBlur(crop_image, (3,3), 0)

# Change color-space from BGR -> HSV
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)

# Create a binary image with where white will be skin colors and rest is black
mask2 = cv2.inRange(hsv, np.array([0,30,80]), np.array([20,255,255]))

# Kernel for morphological transformation    
kernel = np.ones((5,5))

# Apply morphological transformations to filter out the background noise
dilation = cv2.dilate(mask2, kernel, iterations = 1)
erosion = cv2.erode(dilation, kernel, iterations = 1)    
   
# Apply Gaussian Blur and Threshold
filtered = cv2.GaussianBlur(erosion, (3,3), 0)
ret,thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Show threshold image
cv2.imshow("Thresholded", thresh)

# Find contours
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

# Get five largest contour area
cnt = sorted(contours, key = cv2.contourArea, reverse = True)[:5] 

for contour in cnt:    
    # Create bounding rectangle around the contour
    x,y,w,h = cv2.boundingRect(contour)
    if w>100 and h>100:
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)
        
        # Find convex hull
        hull = cv2.convexHull(contour)
        
        # Draw contour
        drawing = np.zeros(crop_image.shape,np.uint8)
        cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
        
        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour,hull)
        
        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger 
        # tips) for all defects
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            
            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image,far,1,[0,0,255],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)

        # Print number of fingers
        if count_defects == 0:
            cv2.putText(img,"1", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX , 1, 2)
        elif count_defects == 1:
            cv2.putText(img,"2", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 2:
            cv2.putText(img, "3", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img,"4", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 4:
            cv2.putText(img,"5", (x+10,y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        else:
            print (count_defects)

# Show required images
cv2.imshow("Result", img)
  
cv2.waitKey(0)
capture.release()
cv2.destroyAllWindows()

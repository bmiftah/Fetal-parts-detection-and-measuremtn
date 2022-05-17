# Code adopted by  from an open source for this particular application April 2022 - Miftah bedru
import cv2 as cv
import numpy as np
import os
import random as rng
def main(image,mask,i):

    rng.seed(56)



    RESULT_DIR = 'Predicted_Ellipse/'


    #print("passed mask ",mask.shape)
    #print("passed image",image.shape)

    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))  # Returns a structuring element of the specified size and shape for morphological operations.
    # smoothing function to remove extra stuff at boundary - perform advanced morphological transformations using an erosion and dilation as basic operations
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se) #  useful in removing noise or adding extra pixel from mask depending on the condition - cv.MORPH_OPEN - in this case
    # canny，80,160
    binary = cv.Canny(mask, 80, 80 * 2)  # ege detection algorithm  80 and 160 - min and max for gradient response to keep .. keep gardient response in the raneg --> 80<G<160
    # 3x3 mask - valued 1
    k = np.ones((3,3), dtype=np.uint8)     # manually specify the structuring element as rect shape using numpy - but we can use built in function like cv.getStructuringElement() -just pass the shape and size of the kernel, you get the desired kernel.
    # expand                               # Like this --> cv.MORPH_RECT,(5,5) , cv.MORPH_ELLIPSE,(5,5) , cv.MORPH_CROSS,(5,5) ..etc
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k) # useful in removing noise k - structuring element or kernel which decides the nature of operation
    # Contour discovery: (image, extract the outermost contour, approximate method of contour) -  find countrs  by joining all the continuous point
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Now get contours - It stores the (x,y) coordinates with same intensity which form the boundary of a shape/line detected by canny and further processed
                                                                                             # it finds white object in black background
    for c in range(len(contours)):    ## loops through the contours  --- contour is here is a python list containing all contours in the our mask  -each contour is numpy array of (x,y) of boundary point of the shape
                                      # In this case we may probably have only one shape and the loop will iterate once and it exit .. because our mask is clearly detected
        # contours can be drawn using cv.drawContours --cv.drawContours(img, contours, -1, (0,255,0), 3) --The below code can draw contour for out mask
        # cv.drawContours(src1, contours, c, (0, 255, 0), 2, 8)

        # print(" Number of contours ", len(contours))



        #  calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of all. returns the rotated rectangle in which the ellipse is inscribed

        (cx, cy), (b, a), angle = cv.fitEllipse(contours[c]) #- cx,cy - center of the ellipse , a - major axis , b- minor axis , angel - angle of rotaton of the ellipse
        # drawing ellipse
        print(" cx = ",cx," cy = ",cy," a = ",a,"b = ",b," angle = ",angle)

        color = (255, 255, 255)
        cv.ellipse(mask, (np.int32(cx), np.int32(cy)),(np.int32((b-2) / 2), np.int32((a-2) / 2)), angle, 0, 360, (255, 255, 255), 2, 7, 0)
        ## ellipse on image
        cv.ellipse(image, (np.int32(cx), np.int32(cy)),(np.int32((b - 2) / 2), np.int32((a - 2) / 2)), angle, 0, 360, (255, 255, 255),2, 7, 0)





        # calculate area - perimeter
        predalc = 2 * (a - b) + np.pi * b #  area= pai*a*b
        predarea = np.pi * (a/2) * (b/2)  # circumference  ellipse area = 2*a*b
        HC = predalc*0.26   #   1 pixel ~ 0.26mm
        # alternative formula for calculating ellipse circumference
       # perimeter = 2*pi*radical((sq(a) + sq(b))/2)
       # HC = 2*pi semi-axis_b+ [ 4 x(semi_axis_a - semi_axis_b)

        # Display area , circumference and other text on the image using PutText - function
        #cv.putText(src, "predalc : " + str(predalc), (20, 15), cv.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        #cv.putText(src, "predarea : " + str(predarea), (20, 35), cv.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        #cv.putText(src, "BPD : " + str(b), (20, 205), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)

        ## caption for mask
        cv.putText(mask,str("Center-(Cx,Cy): ("+str(round(cx,1))+","+str(round(cy,1 ))+" )"),(20, 15), cv.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        cv.putText(mask, str("Major axis a = " + str(round(a,1))) , (20, 180), cv.FONT_HERSHEY_SIMPLEX, .4,(255, 255, 255), 1)
        cv.putText(mask,str( "Minor axis b = " + str(round(b,1))), (20, 200), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv.putText(mask, str("HC = " + str(round(predalc, 1))),(20, 215), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        # caption for image
        cv.putText(image, str("Center-(Cx,Cy): (" + str(round(cx, 1)) + "," + str(round(cy, 1)) + " )"), (20, 15), cv.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        cv.putText(image, str("Major axis a = " + str(round(a, 1))), (20, 180),cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv.putText(image, str("Minor axis b (BPD) = " + str(round(b, 1))), (20, 200),cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv.putText(image, str("HC = " + str(round(predalc, 1))), (20, 215),cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)

        print(" Image ",i,"(cx: ",cx,", cy:",cy,")","(a)/2: ",(a)/2,"(b)/2 ",(b)/2,"angel ",angle)
        print(i," Circumference ",predalc)
        #
        # print(RESULT_DIR)
        # print(str(i))
        # print(src.shape)
        # print(" directory",RESULT_DIR+str(i))
        image_prefix = "img"
        mask_prefix ="mask"
        extension = ".png"
        cv.imwrite(str(RESULT_DIR) + str(mask_prefix) +  str(i) + str(extension), mask) # save mask
        cv.imwrite(str(RESULT_DIR) + str(image_prefix) + str(i) + str(extension), image) # save image
        source_window1 = 'EMP  '
        cv.imshow(source_window1, image)
        cv.waitKey(1000)
        cv.imshow(source_window1, mask)
        #cv.imwrite('C:/HC18/test_set/dsvnet/' + str(i) + '.png', src1)

        cv.waitKey(1000)
        cv.destroyAllWindows()
        return
        #ind=ind+1
if __name__ =="__main__":
    print(" Ellipse fitting module loaded !  ")

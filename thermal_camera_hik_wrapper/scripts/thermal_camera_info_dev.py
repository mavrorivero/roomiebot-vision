#!/usr/bin/python

import cv2
import sys
import imutils
import time
import argparse
import numpy as np
import pytesseract
#import asyncio

from pytesseract import Output

#Color profiles defined in YCrCb
green_ycrcb = np.array([0,255, 0, 121, 0, 130])
yellow_ycrcb = np.array([0,255, 134, 182, 0, 82])
red_ycrcb = np.array([0, 255, 203, 255, 82, 255])

#green_ycrcb = [0,255, 0, 121, 0, 130]
#yellow_ycrcb = [0,255, 134, 182, 0, 82]
#red_ycrcb = [0, 255, 203, 255, 82, 255]

min_percent_area = 0.005
max_percent_area = 0.15

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    redefboxes = []
    for box in boxes:
        abox = np.array((box[0], box[1], box[0]+box[2], box[1]+box[3]))
        redefboxes.append(abox)
    redefboxes = np.asarray(redefboxes)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if redefboxes.dtype.kind == "i":
        redefboxes = redefboxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = redefboxes[:,0]
    y1 = redefboxes[:,1]
    x2 = redefboxes[:,2]
    y2 = redefboxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def get_box_center(box):
    box_center = np.asarray((0.5*(box[2] - box[0]) + box[0], 0.5*(box[3] -box[1])+ box[1]))
    #return box_center
    return box_center[0], box_center[1]


#async def ocr_tesseract_request(img):
#    return await pytesseract.image_to_data(img, output_type=Output.DICT)


#async def ocr_temperature_measures(binary_mask, confidence_wished):
def ocr_temperature_measures(binary_mask, confidence_wished):
    mask_inverse = np.invert(binary_mask)
    #cv2.imwrite("green_test.png", mask_inverse)
    custom_oem_psm_config = r'--oem 3 --psm 3'
    st1 = time.time()
    ocr_result = pytesseract.image_to_data(mask_inverse, output_type=Output.DICT,
        config=custom_oem_psm_config)
    #ocr_result = await ocr_tesseract_request(mask_inverse)
    st2 = time.time()
    print("Time consumed in extract ocr: ", st2 - st1)

    st1 = time.time()
    measures = []
    if len(ocr_result) <= 0:
        return np.asarray(measures)
    else:
        for i in range(0, len(ocr_result["text"])):
            conf = int(ocr_result["conf"][i])
            text = ocr_result["text"][i]

            if conf > confidence_wished and len(text) != 0:
                #x = ocr_result["left"][i]
                #y = ocr_result["top"][i]
                #w = ocr_result["width"][i]
                #h = ocr_result["height"][i]
                #box = [x,y,x+w,y+h]
                box = [ocr_result["left"][i], ocr_result["top"][i], 
                    ocr_result["left"][i] + ocr_result["width"][i], 
                    ocr_result["top"][i] + ocr_result["height"][i]]

                cX, cY = get_box_center(box)

                measures.append((text, box, cX, cY))
                #print("confidence: {}".format(conf))
                #print("Text: {}".format(text))
                #print("BoundingBox: ", box)
                #print("")
    st2 = time.time()
    print("Time consumed in get temperature measures in rutine: ", st2 - st1)
    return measures


def relate_measures(targets, measures):
    mesBoxes = []
    for box in measures:
        mesBoxes.append(box[1])

    relations = []
    if len(mesBoxes) == 1 and len(targets)==1:
        relations.append((measures[0][:2], targets[0]))
        return relations
    else:
        print("Do intent of kmeans with scitkit images")
    return relations


def find_rois(img, min_percent, max_percent):
    cnts = cv2.findContours(img, cv2.RETR_LIST,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    boxes = []
    found = False
    
    imgArea = img.shape[0]*img.shape[1]
    minArea = int(min_percent*imgArea)
    maxArea = int(max_percent*imgArea)
    #print("Image area: ", imgArea)
    #print("."*30)
    #print("Area minima de referencia: ", minArea)
    #print("."*30)
    #print("Area maxima de referencia: ", maxArea)
    #print("."*30)
    #aimg = img.copy()
    #aimg = cv2.cvtColor(aimg, cv2.COLOR_GRAY2BGR)

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            try:
                mom = cv2.moments(c)
                #mbox = cv2.boundingRect(c)
                #cv2.rectangle(aimg, (mbox[0],mbox[1]), (mbox[0]+mbox[2], mbox[1]+mbox[3]), (0,0,255), 2)
                #cv2.imshow("Show each contour found without any kind of selection", aimg)
                #print("Area: ", mom['m00'], "\t", "Total pts: ", len(c), "\tBox[x,y,w,h]: ", mbox)
                #cv2.waitKey()

                if minArea < mom['m00'] and mom['m00'] < maxArea:
                    box = cv2.boundingRect(c)
                    #cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,0), 2)
                    #cv2.imshow("Debug", img)
                    #cv2.waitKey()
                    boxes.append(box)
                    found = True
                    #print("Area: ", mom['m00'], "\t", "Total pts: ", len(c) , "\tBox[x,y,w,h]: ", box)
            except:
                #print("Not found anything")
                continue
    return found, boxes


def getMask(img, pfColor):
    lowLimits = (pfColor[0],pfColor[2],pfColor[4])
    highLimits = (pfColor[1],pfColor[3],pfColor[5])

    mask = cv2.inRange(img, lowLimits, highLimits)
    return mask


def draw_bbs(boxes, img, color):
    for box in boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
    
    cv2.imshow("Original Image with all bounding boxes found", img)
    cv2.waitKey()
    cv2.destroyWindow("Original Image with all bounding boxes found")


def wrapper_callback(img_src):
    yuv_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2YCR_CB)
    
    #get binary mask to each target
    st0 = time.time()
    normal_measure_mask = getMask(yuv_img, green_ycrcb)
    warning_measure_mask = getMask(yuv_img, yellow_ycrcb)
    alert_measure_mask = getMask(yuv_img, red_ycrcb)
    st1 = time.time()
    print("Time consumed in extract binary masks: ", st1-st0)

    are_normals, normalTargets = find_rois(normal_measure_mask,  min_percent_area, max_percent_area)
    are_warnings, warningTargets = find_rois(warning_measure_mask, min_percent_area, max_percent_area)
    st2 = time.time()
    are_alerts, alertTargets = find_rois(alert_measure_mask, min_percent_area, max_percent_area)
    st3 = time.time()
    print("Time consumed in find target rois: ", st3-st2)

    if are_alerts:
        print("*"*40)
        print("Person(s) Detected with fever :( ... ")
        st4 = time.time()
        alertTargets = non_max_suppression_fast(np.asarray(alertTargets), 0.5)
        st5 = time.time()
        print("Time consumed in non_max_supression: ", st5-st4)
        #print(alertTargets)
        draw_bbs(alertTargets, img_src, (0, 0, 0))
        print("*"*40)
        st6 = time.time()
        alert_measures = ocr_temperature_measures(alert_measure_mask, 70)
        st7 = time.time()
        print("Time consumed in get ocr temperature measure: ", st7-st6)
        #print("breakpoint")
        lectures = relate_measures(alertTargets, alert_measures)
        print("Total targets found: ", len(alertTargets), "\twith a temperature of: ", lectures)

    if are_warnings:
        print("-"*30)
        print("Person(s) Detected probably fever -_- ... ")
        warningTargets = non_max_suppression_fast(np.asarray(warningTargets), 0.5)
        draw_bbs(warningTargets, img_src, (255, 0, 0))
        st6 = time.time()
        warning_measures = ocr_temperature_measures(warning_measure_mask, 50)
        st7 = time.time()
        print("Time consumed in get ocr temperature measure: ", st7-st6)
        lectures = relate_measures(warningTargets, warning_measures)
        print("Total targets found: ", len(warningTargets), "\twith a temperature of: ", lectures)
        print("-"*40)
    if are_normals:
        print("-"*30)
        print("Person with normal Temperature :)")
        normalTargets = non_max_suppression_fast(np.asarray(normalTargets), 0.5)
        draw_bbs(normalTargets, img_src, (0, 255, 255))
        #st6 = time.time()
        normal_measures = ocr_temperature_measures(normal_measure_mask, 50)
        #st7 = time.time()
        #print("Time consumed in get ocr temperature measure: ", st7-st6)
        #lectures = relate_measures(normalTargets, normal_measures)
        #print("Total targets found: ", len(normalTargets), "\twith a temperature of: ", lectures)
        print("Total targets found: ", len(normalTargets))
        print("-"*40)
    
    ''' To Save testing images
    key = cv2.waitKey()
    if key & 0xFF == ord("s"):
        count_img+=1
        img_name = "testImg"+str(count_img)+".png"
        cv2.imwrite(img_name, img_src)
        print("Image saved")
    #'''
    #To Debug
    #cv2.imshow("Streaming video", img_src)
    #cv2.imshow("Normal measure binary mask", normal_measure_mask)
    #cv2.imshow("Warning measure binary mask", warning_measure_mask)
    #cv2.imshow("Alert measure binary mask", alert_measure_mask)
    #cv2.waitKey(1)

def main(args):
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="Specify the image to work",
        default="imgs/testImg0.png")
    args = vars(ap.parse_args())
    print("."*60)
    print("Started thermal camera info script wrapper...")
    print("."*60)
    img = cv2.imread(args["image"])
    wrapper_callback(img)

    cv2.destroyAllWindows()
    print("*"*60)
    print("Thermal_camera_wrapper node finished ...")

if __name__ == "__main__":
    main(sys.argv)
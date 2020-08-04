#!/usr/bin/python

import cv2
import sys
import time
import rospy
import imutils
import numpy as np
import pytesseract

from pytesseract import Output
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

#Color profiles defined in YCrCb
green_ycrcb = np.array([0,255, 0, 121, 0, 130])
yellow_ycrcb = np.array([0,255, 134, 182, 0, 82])
red_ycrcb = np.array([0, 255, 203, 255, 82, 255])
#BGR profiles color to draw the three differentes targets
pink_bgr = (153, 51, 255)
cyan_bgr = (255, 255, 102)
dark_green = (0, 102, 51)
#To save images test from streaming
#count_img = 3
#Percent area parameter to filter rois
min_percent_area = 0.005
max_percent_area = 0.15
#Limit of points that build a rectangular contour
# considering image of 640x480
maxPtsRect = 20
#Allowed overlap to non maximum supression
nonMaxSupOverlap = 45

thermal_stream = rospy.get_param("/thermal_camera_stream_topic", "/hik_cam_node/hik_camera")

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


def ocr_temperature_measures(binary_mask, confidence_wished):
    mask_inverse = np.invert(binary_mask) #invert binary mask to get ocr
    measures = []
    try:
        ocr_result = pytesseract.image_to_data(mask_inverse, output_type=Output.DICT)

        if len(ocr_result) <= 0:
            return np.asarray(measures)
        else:
            for i in range(0, len(ocr_result["text"])):
                conf = int(ocr_result["conf"][i])
                text = ocr_result["text"][i]

                if conf >= confidence_wished and len(text) != 0:
                    x = ocr_result["left"][i]
                    y = ocr_result["top"][i]
                    w = ocr_result["width"][i]
                    h = ocr_result["height"][i]
                    box = [x, y, x+w, y+h]
                    #box = [ocr_result["left"][i], ocr_result["top"][i], 
                    #    ocr_result["left"][i] + ocr_result["width"][i], 
                    #    ocr_result["top"][i] + ocr_result["height"][i]]

                    cX, cY = get_box_center(box)

                    measures.append((text, box, cX, cY))
                    #print("confidence: {}".format(conf))
                    #print("Text: {}".format(text))
                    #print("BoundingBox: ", box)
                    #print("")
        return measures
    except:
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
        print("To Do intent of kmeans with scikit image")
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

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            try:
                mom = cv2.moments(c)
                #print("Area: ", mom['m00'])
                if minArea < mom['m00'] and mom['m00'] < maxArea and len(c) <= maxPtsRect:
                    box = cv2.boundingRect(c)
                    boxes.append(box)
                    found = True
                    #print("Area: ", mom['m00'], "\t", "Total pts: ", len(c) , "Box[x,y,w,h]: ", box)
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
    cv2.waitKey(1)
    #cv2.destroyWindow("Original Image with all bounding boxes found")




def wrapper_callback(data):
    #global count_img
    #start = time.time()

    bridge = CvBridge()
    img_src = bridge.imgmsg_to_cv2(data, "bgr8")
    yuv_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2YCR_CB)
    
    #get binary mask to each target
    normal_measure_mask = getMask(yuv_img, green_ycrcb)
    warning_measure_mask = getMask(yuv_img, yellow_ycrcb)
    alert_measure_mask = getMask(yuv_img, red_ycrcb)

    are_normals, normalTargets = find_rois(normal_measure_mask, min_percent_area, 1.5*max_percent_area)
    are_warnings, warningTargets = find_rois(warning_measure_mask, min_percent_area, 1.5*max_percent_area)
    are_alerts, alertTargets = find_rois(alert_measure_mask, min_percent_area, 1.5*max_percent_area)

    if are_alerts:
        print("+"*40)
        #print("Person(s) Detected with fever :( ... ")
        alertTargets = non_max_suppression_fast(np.asarray(alertTargets), nonMaxSupOverlap)
        draw_bbs(alertTargets, img_src, pink_bgr) #pink
        #alert_measures = ocr_temperature_measures(alert_measure_mask, 20)
        #lectures = relate_measures(alertTargets, alert_measures)
        #end = time.time()
        #time_consumed = end - start
        #print("Total targets found: ", len(alertTargets), "\twith a temperature of: ", lectures)
        print("Total targets found: ", len(alertTargets))
        #print("Time consumed of th request: ", time_consumed)
        print("+"*40)
    
    if are_warnings:
        print("*"*40)
        #print("Person(s) Detected probably fever -_- ... ")
        warningTargets = non_max_suppression_fast(np.asarray(warningTargets), nonMaxSupOverlap)
        draw_bbs(warningTargets, img_src, cyan_bgr) #cyan
        #warning_measures = ocr_temperature_measures(warning_measure_mask, 20)
        #lectures = relate_measures(warningTargets, warning_measures)
        #end = time.time()
        #time_consumed = end - start
        #print("Total warning targets found: ", len(warningTargets), "\twith a temperature of: ", lectures)
        print("Total warning targets found: ", len(warningTargets))
        #print("Time consumed of th request: ", time_consumed)
        print("*"*40)
    
    if are_normals:
        print("="*40)
        #print("Person with normal Temperature :) ... yei")
        normalTargets = non_max_suppression_fast(np.asarray(normalTargets), nonMaxSupOverlap)
        draw_bbs(normalTargets, img_src, dark_green) #a different green
        #normal_measures = ocr_temperature_measures(normal_measure_mask, 20)
        #lectures = relate_measures(normalTargets, normal_measures)
        #end = time.time()
        #time_consumed = end - start
        #print("Total normal targets found: ", len(normalTargets), "\twith a temperature of: ", lectures)
        print("Total normal targets found: ", len(normalTargets))
        #print("Time consumed of th request: ", time_consumed)
        #print(normalTargets)
        print("="*40)
    
    ''' To Save testing images
    key = cv2.waitKey()
    if key & 0xFF == ord("s"):
        count_img+=1
        img_name = "testImg"+str(count_img)+".png"
        cv2.imwrite(img_name, img_src)
        print("Image saved")
    #'''


def activation_node_callback(data):
    global active_node
    global img
    
    active_node = data.data
    
    if active_node:
        img = rospy.Subscriber(thermal_stream, Image, wrapper_callback)
    else:
        try:
            img.unregister()
            print("Disable try")
        except:
            print("Disable except")
            pass


def main(args):
    global img
    rospy.init_node("thermal_camera_wrapper_node", anonymous=False)
    print("."*60)
    print("Started thermal camera info script wrapper...")
    print("."*60)
   
    enable_node = rospy.get_param("/enable_search_humans_by_temp", "/roomie_cv/hik_camera/enable_search_by_temperature")

    rospy.Subscriber(enable_node, Bool, activation_node_callback)
    img = rospy.Subscriber(thermal_stream, Image, wrapper_callback)
    img.unregister()
    
    while not rospy.is_shutdown():
        rospy.spin()
    
    cv2.destroyAllWindows()
    print("*"*60)
    print("Thermal_camera_wrapper node finished ...")

if __name__ == "__main__":
    main(sys.argv)
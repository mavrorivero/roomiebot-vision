#!/usr/bin/python

import cv2
import sys
import time
import rospy
import imutils
import numpy as np
import random
import rospkg

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from roomie_cv_msg.msg import CVThermalObject
from roomie_cv_msg.msg import CVThermalObjects


# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()


#Color profiles defined in YCrCb
green_ycrcb = [0, 255, 0, 121, 0, 130]
yellow_ycrcb = [0, 255, 134, 182, 0, 82]
red_ycrcb = [0, 255, 203, 255, 82, 255]
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
#Allowed percent overlap to non maximum supression
nonMaxSupOverlap = 45

#Get topic of thermal camera streaming
#thermal_stream = rospy.get_param("/thermal_camera_stream_topic", "/hik_cam_node/hik_camera")
thermal_stream = rospy.get_param("/thermal_camera_stream_topic", "/hik2/hik_cam_node/hik_thermal_cam_ds2t/image_raw")
#Topic for publish results
pub_targets = rospy.Publisher("roomie_cv/thermal_targets_found", CVThermalObjects, queue_size=1)

success_count = 0
frame_count = 0
success_pub_count = 0
frame_pub_count = 0

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
    overlapThresh = overlapThresh/100.0
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


def get_binary_mask(img, pfColor):
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


def prepare_result_to_publish(img, targets, target_type, color):
    global success_pub_count
    global frame_pub_count
    global results

    for target in targets:
        object_detected = CVThermalObject()
        object_detected.label = target_type
        
        #Upload color profile
        for i in range(0, 6):
            #print color[i]
            object_detected.color_yCrCb[i] = color[i]
        
        #Upload boundingbox data
        for i in range(0, 4):
            object_detected.bounding_box[i] = target[i]
        
        results.objects.append(object_detected)


def wrapper_callback(data):
    global success_count
    global frame_count
    global results
    
    #Array of objects create like ros msg to publish results
    results = CVThermalObjects()
    #rate = rospy.Rate(5)
    img_src = cv2.imread(data)
    cv2.imshow("Original Image with all bounding boxes found", img_src)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    bridge = CvBridge()
    #img_src = bridge.imgmsg_to_cv2(data, "bgr8")
    yuv_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2YCR_CB)
    frame_count+=1
    
    #get binary mask to each target
    normal_measure_mask = get_binary_mask(yuv_img, green_ycrcb)
    warning_measure_mask = get_binary_mask(yuv_img, yellow_ycrcb)
    alert_measure_mask = get_binary_mask(yuv_img, red_ycrcb)

    are_normals, normalTargets = find_rois(normal_measure_mask, 
        min_percent_area, 1.5*max_percent_area)
    are_warnings, warningTargets = find_rois(warning_measure_mask, 
        min_percent_area, 1.5*max_percent_area)
    are_alerts, alertTargets = find_rois(alert_measure_mask, 
        min_percent_area, 1.5*max_percent_area)

    if are_alerts:
        print("+"*40)
        success_count+=1
        print("Person(s) Detected with fever :( ... ")
        alertTargets = non_max_suppression_fast(np.asarray(alertTargets), nonMaxSupOverlap)
        prepare_result_to_publish(img_src, alertTargets, "fever", red_ycrcb)
        #draw_bbs(alertTargets, img_src, pink_bgr) #pink
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
        success_count+=1
        print("Person(s) Detected probably fever -_- ... ")
        warningTargets = non_max_suppression_fast(np.asarray(warningTargets), nonMaxSupOverlap)
        #draw_bbs(warningTargets, img_src, cyan_bgr) #cyan
        prepare_result_to_publish(img_src, warningTargets, "warning", yellow_ycrcb)
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
        success_count+=1
        print("Person with normal Temperature :) ... yei")
        normalTargets = non_max_suppression_fast(np.asarray(normalTargets), nonMaxSupOverlap)
        #draw_bbs(normalTargets, img_src, dark_green) #a different green
        prepare_result_to_publish(img_src, normalTargets, "normal", green_ycrcb)
        #normal_measures = ocr_temperature_measures(normal_measure_mask, 20)
        #lectures = relate_measures(normalTargets, normal_measures)
        #end = time.time()
        #time_consumed = end - start
        #print("Total normal targets found: ", len(normalTargets), "\twith a temperature of: ", lectures)
        print("Total normal targets found: ", len(normalTargets))
        #print("Time consumed of th request: ", time_consumed)
        #print(normalTargets)
        print("="*40)
    
    if are_normals or are_warnings or are_alerts:
        results.frame = bridge.cv2_to_imgmsg(img_src, "bgr8")
        results.frame.width = img_src.shape[1]
        results.frame.height = img_src.shape[0]
        results.frame.header.seq = success_count
        results.frame.header.frame_id = str(success_count)
        results.frame.header.stamp = rospy.Time.now()
        
        results.header.stamp = rospy.Time.now()
        results.header.seq = success_count
        results.header.frame_id = str(frame_count)
    
    else:
        results.header.stamp = rospy.Time.now()
        results.header.seq = frame_count
        results.header.frame_id = str(frame_count)
        results.objects = []
    
    pub_targets.publish(results)
    #rate.sleep()

    
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
    global rospack
    
    active_node = data.data
    
    if active_node:
        rospy.loginfo("*"*40)
        rospy.loginfo("Node Enabled :)")
        #Get random image from ../imgs
        #img = rospy.Subscriber(thermal_stream, Image, wrapper_callback)
        # get the file path for rospy_tutorials
        path = rospack.get_path('thermal_camera_hik_wrapper')
        imgs_list= ['0', '2', '3']        
        filename = path+'/scripts/imgs/testImg'+str(random.choice(imgs_list))+'.png'
        rospy.loginfo('filename: '+str(filename))
        wrapper_callback(filename)
    else:
        #img.unregister()
        rospy.loginfo("*"*40)
        rospy.loginfo("Node Disabled 0_0")


def main(args):
    global img
    
    rospy.init_node("thermal_camera_wrapper_node", anonymous=False)
    rospy.loginfo("."*60)
    rospy.loginfo(" Started thermal camera info script wrapper...")
    rospy.loginfo("  by NachoBot 0_0")
    rospy.loginfo("."*60)
   
    enable_node = rospy.get_param("/enable_search_humans_by_temp", "/roomie_cv/enable_search_by_temperature")

    rospy.Subscriber(enable_node, Bool, activation_node_callback)
    #img = rospy.Subscriber(thermal_stream, Image, wrapper_callback)
    #img.unregister()
    
    while not rospy.is_shutdown():
        rospy.spin()
    
    cv2.destroyAllWindows()
    rospy.loginfo("*"*60)
    rospy.loginfo("Thermal_camera_wrapper node finished ...")

if __name__ == "__main__":
    main(sys.argv)
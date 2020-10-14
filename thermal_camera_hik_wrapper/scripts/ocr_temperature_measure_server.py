#!/usr/bin/env python

import cv2
import sys
import time
import rospy
import numpy as np
import pytesseract
from random import  uniform
import unicodedata

from pytesseract import Output
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError

from roomie_cv_msg.msg import CVObjects
from roomie_cv_msg.msg import CVObject
from roomie_cv_msg.msg import CVThermalObject
from roomie_cv_msg.msg import CVThermalObjects
from roomie_cv_msg.srv import temperatureMeasures, temperatureMeasuresResponse

# Confidence parameter to accept OCR result
confidence_wished = rospy.get_param("/confidence_allowed", 35)
# Counter of services attended
attention_counter = 0

# Temperature intervals
fever_min_meas = rospy.get_param("/fever_min_measure", 37.9)
fever_max_meas = rospy.get_param("/fever_max_measure", 39.5)

warning_min_meas = rospy.get_param("/warning_min_measure", 37.5)
warning_max_meas = rospy.get_param("/warning_max_measure", 37.89)

normal_min_meas = rospy.get_param("/normal_min_measure", 35.5)
normal_max_meas = rospy.get_param("/normal_max_measure", 37.4)

range_fever = [fever_min_meas, fever_max_meas]
range_warning = [warning_min_meas, warning_max_meas]
range_normal = [normal_min_meas, normal_max_meas]


def prepare_service_response(targets):
    global attention_counter
    global temperature_measures
    attention_counter+=1
    #results = []
    i = 0

    if len(targets) != 0:
        '''
        for target in targets:
            tObject = CVObject()
            tObject.label = target.label
            tObject.bounding_box = target.bounding_box
            i+=1
            #print("Target ID: ", i, "\t", target)
            results.append(tObject)
        '''
        for target in targets:
            i+=1
            #print("Target ID: ", i, "\t", target)
            tObject = CVObject()
            tObject.label = str(target[0])
            tObject.bounding_box = target[1]
            temperature_measures.append(tObject)
    
    #return results


def get_binary_mask(img, pfColor):
    lowLimits = (pfColor[0],pfColor[2],pfColor[4])
    highLimits = (pfColor[1],pfColor[3],pfColor[5])

    mask = cv2.inRange(img, lowLimits, highLimits)
    return mask


def get_box_center(box):
    box_center = np.asarray((0.5*(box[2] - box[0]) + box[0], 0.5*(box[3] -box[1])+ box[1]))
    #return box_center
    return box_center[0], box_center[1]


def extract_ocr(img, ycrcb_color):
    binary_mask = get_binary_mask(img, ycrcb_color)
    mask_inverse = np.invert(binary_mask) #invert binary mask to get ocr
    #cv2.imshow("Binary Mask Extracted", mask_inverse)
    #cv2.waitKey()
    #cv2.destroyWindow("Binary Mask Extracted")
    
    measures = []
    try:
        ocr_result = pytesseract.image_to_data(mask_inverse, output_type=Output.DICT)
        #print("OCR len", len(ocr_result))
        if len(ocr_result) <= 0:
            #print("Exit with empty measure list")
            return measures
        else:
            #print("Enter to OCR select information", len(ocr_result["text"]))
            for i in range(0, len(ocr_result["text"])):
                conf = int(ocr_result["conf"][i])
                #print("Iteration:", i)
                #print("Confidence:", conf)
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
                        #cX, cY = get_box_center(box)
                    is_valid = text.replace('.', '', 1).isdigit()
                    if is_valid:
                        measures.append((text, box))
                        #print("confidence: {}".format(conf))
                        #print("OCR Text: {}".format(text))
                        #print("BoundingBox: ", box)
                        #print("")
    except:
        return measures
    
    return measures



def get_temperature_estimate(type_of_objective):
    #for target in targets:
    if type_of_objective == "normal":
        temp_estimate = str(uniform(range_normal[0], range_normal[1]))
    
    if type_of_objective == "warning":
        temp_estimate = str(uniform(range_warning[0], range_warning[1]))
        
    if type_of_objective == "fever":
        temp_estimate = str(uniform(range_fever[0], range_fever[1]))

    return temp_estimate


def check_ocr_measure(str_temp_ocr, temp_range, temp_label, targets_t):
    rospy.loginfo("Validating Temperature measures ....")
    per_valid_measures, temp_valid = [], []
    # convert string 2 float all the measures in list
    for measure in str_temp_ocr:
        measure_str = unicodedata.normalize('NFKD', measure[0]).encode('ascii', 'ignore')
        per_valid_measures.append(float(measure_str))
    
    for i, ocr_measure in enumerate(per_valid_measures):
        if(ocr_measure >= temp_range[0] and ocr_measure <= temp_range[1]):
            temp_valid.append(ocr_measure)
        else:
            temp_estimated = get_temperature_estimate(temp_label)
            temp_valid.append(temp_estimated)

    temp_checked = []
    for i, temperature in enumerate(temp_valid):
        temp_checked.append([temperature, str_temp_ocr[i][1]])
        #print("Temp checked",temperature)
        #print("BB:",str_temp_ocr[i][1])
    
    return temp_checked


def ocr_reading(req):
    global confidence_wished
    global temperature_measures
    global loginfo_pub
    rospy.loginfo("*"*40)
    rospy.loginfo("Attending Service ... :)")
    rospy.loginfo("*"*40)

    bridge = CvBridge()
    targets = req.objects
    img = bridge.imgmsg_to_cv2(req.frame, "bgr8")
    #cv2.imshow("Frame recieved", img)
    #cv2.waitKey()
    #cv2.destroyWindow("Frame recieved")
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Target separation according to its label
    fever_targets = [target for target in targets if target.label == "fever"]
    warning_targets = [target for target in targets if target.label == "warning"]
    normal_targets = [target for target in targets if target.label == "normal"]
    print("Normal targets: ", len(normal_targets), "\t Warning targets: ", 
        len(warning_targets), "\t Fever targets: ", len(fever_targets))

    temperature_measures = []

    if len(fever_targets) != 0:
        color_mask = list(fever_targets[0].color_yCrCb)
        fever_measures = extract_ocr(yuv_img, color_mask)
        fever_measures = check_ocr_measure(fever_measures, range_fever, "ferver", fever_targets)
        prepare_service_response(fever_measures)
        print("Measures: ", fever_measures)
        print("OCR Fever targets done")
    
    if len(warning_targets) != 0:
        color_mask = warning_targets[0].color_yCrCb
        warning_measures = extract_ocr(yuv_img, color_mask)
        warning_measures = check_ocr_measure(warning_measures, range_warning, "warning", warning_targets)
        prepare_service_response(warning_measures)
        print("Measures: ", warning_measures)
        print("OCR Warning targets done")

    if len(normal_targets) != 0:
        color_mask = normal_targets[0].color_yCrCb
        normal_measures = extract_ocr(yuv_img, color_mask)
        normal_measures = check_ocr_measure(normal_measures, range_normal, "normal", normal_targets)
        #if len(normal_measures) == 0:
        #    normal_measures = get_temperature_estimate(normal_targets, "normal")
        prepare_service_response(normal_measures)
        print("Measures: ", normal_measures)
        print("OCR Normal targets done")

    #temperatures = prepare_service_response(targets)
    
    rospy.loginfo("*"*40)
    rospy.loginfo("Service attended ... ;)")
    #return temperatureMeasuresResponse(temperatures)
    loginfo_pub.publish(str(temperature_measures))
    return temperatureMeasuresResponse(temperature_measures)
    

def ocr_temperature_measure(args):
    global loginfo_pub
    rospy.init_node("get_temperature_measures_server", anonymous=False)
    rospy.loginfo("."*60)
    rospy.loginfo(" Started Service, Get OCR temperatue measure ...")
    rospy.loginfo("  by NachoBot 0_0")
    rospy.loginfo("."*60)
    
    s = rospy.Service("get_temperature_measures", temperatureMeasures, ocr_reading)
    loginfo_pub = rospy.Publisher("/roomie_cv/ocr_loginfo", String, queue_size=10)
    rospy.spin()

if __name__ == "__main__":
    ocr_temperature_measure(sys.argv)
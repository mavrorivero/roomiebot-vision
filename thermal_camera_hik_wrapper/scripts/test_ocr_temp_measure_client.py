#!/usr/bin/env python

'''
Este cliente debe de mandar la informacion necesaria al servicio, para obtener
la medicion de temperatura de cada uno de los objetivos encontrados
Para esto, el cliente se debe de suscribir al topic de resultados, agarrar un frame
de todo el msg resultado y mandarlo al servicio
'''

import cv2
import sys
import time
import rospy
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError

from roomie_cv_msg.msg import CVThermalObject
from roomie_cv_msg.msg import CVThermalObjects
from roomie_cv_msg.srv import temperatureMeasures, temperatureMeasuresResponse

thermal_targets = rospy.get_param("/roomie_cv/thermal_targets_found", "/roomie_cv/thermal_targets_found")

def activation_node_callback(flag):
    global active_node
    global thTargets

    active_node = flag.data

    if active_node:
        print("*"*40)
        print("Do client request :)")
        #thTargets = rospy.Subscriber(thermal_targets, CVThermalObjects, get_objects_detected)
        flag_catch_msg = True
        try:
            while flag_catch_msg:
                thTargets = rospy.wait_for_message(thermal_targets, CVThermalObjects, timeout=10)
                if len(thTargets.objects) != 0:
                    flag_catch_msg = False
                    print('Targets cached ...', len(thTargets.objects))
            
            rospy.wait_for_service("get_temperature_measures")
            
            try:
                request_client = rospy.ServiceProxy("get_temperature_measures", temperatureMeasures)
                resp = request_client(thTargets.objects, thTargets.frame)
                #print(resp)
                print("*"*40)
                print("Client request attended ...")
            except rospy.ServiceException:
                print("Did not get response of service :(")
        
        except rospy.ROSException:
            print("*"*40)
            print("Did not get response of service :(, No answer of thermal_objects_found topic")
                
    else:
        print("*"*40)
        #thTargets.unregister()
        print("Wait client request 0_0")


def test_ocr_temp_measure_client(args):
    print("."*60)
    print(" Started testing client of OCR temperature measure ...")
    print("  by NachoBot 0_0")
    print("."*60)

    global thTargets

    rospy.init_node("testing_node_for_thermalCamera", anonymous=False)
    
    enable_node = rospy.get_param("/do_client_test_request", "/roomie_cv/do_client_test_request")
    #thTargets = rospy.Subscriber(thermal_targets, CVThermalObjects, get_objects_detected)
    rospy.Subscriber(enable_node, Bool, activation_node_callback)
    #thTargets.unregister()

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    test_ocr_temp_measure_client(sys.argv)
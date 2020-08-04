> **Thermal Camera HIK Wrapper**

This package have 1 node that get the number of targets detected in the HIK Vision Thermal camera streaming, and them are classify in three categories: "normal, "warning" and "fever". This classes correspond to the categories that handle the streaming results of the camera DS2Txxxxxx, defined three differente colors to each category of target found.

It also has a OCR service node, that could return the temperature measure of all the targets that appear in one frame. In case the OCR does not get a result, I get and random controlled measure considering each of the temperature ranges, into which the three categories of targets detected by the camera are divided.

> *Library Dependencies*

This package use the follow differente and specific libraries: pytesseract and imutils. With the next command lines, you cand install this libraries:
```
    pip install pytesseract
```

```
    pip install imutils
```

Finally, this package use **roomie_cv_msg** package that contain all the specific messages and service used in both nodes.


> About thermal_camera_info_node.py

This node **subscribes** to the following topics:
```
    /roomie_cv/enable_search_by_temperature
```
and
```
    /hik_cam_node/hik_camera
```

And you can changed using the follow **rosparam** in a launch file:
```
    /enable_search_humans_by_temp
```
and
```
    /thermal_camera_stream_topic
```

To Active the functionality of this node, you have to publish a Boolean  flag in the topic `/roomie_cv/enable_search_by_temperature` or whatever has been selected with the rosparam `/enable_search_humans_by_temp`, with the value of True.

And for Desactive this node, publish a False in the mentionated topic.

**Publishers**

The results of all the targets found are published in 
```
    roomie_cv/thermal_targets_found
```
using the msg CVThermalObjects.


> About ocr_temperature_measure_server.py

For use this service node, you have to **require** it with two variables:

    CVThermalObject[] object
    
    sensor_msgs/Image frame

and the **answer** of the service is a:

    CVObject[] rois

You could check the `test_ocr_temp_measure_client.py` as an example to use the service node.
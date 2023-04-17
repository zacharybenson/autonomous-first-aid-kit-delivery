import cv2
import numpy as np

# realsesne camera
import pyrealsense2.pyrealsense2 as rs

"""
use custom yolo to evaluate video stream
"""

CONF_THRESH, NMS_THRESH = 0.05,0.3

def detect_annotate(img, net, classes):

    cv2.putText(img, 'detecting...', (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (192, 192), swapRB=False, crop=False)

    #blob = cv2.dnn.blobFromImage(
    #    cv2.resize(img, (416, 416)),
    #    0.007843, (416, 416), 127.5)

    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESH:
                center_x, center_y, w, h = \
                    (detection[0:4] * np.array([frame_w, frame_h, frame_w, frame_h])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
            
    if len(b_boxes) > 0:
        # Perform non maximum suppression for the bounding boxes
        # to filter overlapping and low confidence bounding boxes.
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
        for index in indices:
            x, y, w, h = b_boxes[index]
            cv2.rectangle(img, (x, y), (x + w, y + h), (20,20,230), 2)
            cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, myColor, 2)


if __name__ == '__main__':

    in_weights = 'yolov4-tiny-custom_last.weights'
    in_config = 'yolov4-tiny-custom.cfg'
    name_file = 'custom.names'

    """
    load names
    """
    with open(name_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print(classes)

    """
    Load the network
    """
    net = cv2.dnn.readNetFromDarknet(in_config, in_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    """
    iminitalize video from realsense
    """

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    profile = pipeline.get_active_profile()
    image_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    image_intrinsics = image_profile.get_intrinsics()
    frame_w, frame_h = image_intrinsics.width, image_intrinsics.height

    print('image: {} w  x {} h pixels'.format(frame_w, frame_h))


    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    myColor = (20,20,230)

    """
    run main loop
    """
    while True:
        timer = cv2.getTickCount()

        frameset = pipeline.wait_for_frames()
        frame = frameset.get_color_frame()
        if not frame:
            print('missed frame...')
            continue
        img = np.asanyarray(frame.get_data())

        detect_annotate(img, net, classes)
                
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        
        myColor = (20,20,230)
        cv2.putText(img, '{:.0f} fps'.format(fps), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)
        cv2.imshow("Tracking", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


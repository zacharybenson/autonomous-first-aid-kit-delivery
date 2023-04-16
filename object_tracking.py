import cv2
import pyrealsense2.pyrealsense2 as rs
import numpy as np

# visdrone stuff
output_layers = None
visdrone_net = None
visdrone_classes = []
CONF_THRESH, NMS_THRESH = 0.05, 0.5

# Configure realsense camera stream
pipeline = rs.pipeline()
config = rs.config()

# x,y center for 640x480 camera resolution.
FRAME_HORIZONTAL_CENTER = int(320)
FRAME_VERTICAL_CENTER = int(240)
FRAME_HEIGHT = int(480)
FRAME_WIDTH = int(640)

rnd_background = np.random.randint(0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3)).astype('uint8')

total_track_misses = 0
TRACKER_MISSES_MAX = 6
confirmed_object_tracking = False

tracker = None
DEFAULT_TRACKER_TYPE = 'CSRT'
cv_version = cv2.__version__


def create_tracker(tracker_type='CSRT'):
    global tracker

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker


def load_visdrone_network():

    global visdrone_net, output_layers, visdrone_classes
    # TODO: be sure to set these paths correctly, according to your system.
    in_weights = '/home/usafa/PycharmProjects/USAFA/Drone001/yolo_visdron/yolov4-tiny-custom_last.weights'
    in_config = '/home/usafa/PycharmProjects/USAFA/Drone001/yolo_visdron/yolov4-tiny-custom.cfg'
    name_file = '/home/usafa/PycharmProjects/USAFA/Drone001/yolo_visdron/custom.names'


    """
    load names
    """
    with open(name_file, "r") as f:
        visdrone_classes = [line.strip() for line in f.readlines()]

    """
    Load the network
    """
    visdrone_net = cv2.dnn.readNetFromDarknet(in_config, in_weights)
    visdrone_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    visdrone_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layers = visdrone_net.getLayerNames()

    out_layers = visdrone_net.getUnconnectedOutLayers()
    if len(out_layers.shape) == 1:
        output_layers = [layers[i - 1] for i in out_layers]
    else:
        output_layers = [layers[i[0] - 1] for i in out_layers]


def confirm_obj_in_bbox(frame, bbox, frame_write=None, show_img=False, in_debug=False):
    try:

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            return False

        # Widen our margin around
        # the tight object bounding box
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cr = int(max(w, h) / 2)

        # Use random noise background
        blank_image = rnd_background.copy()

        # Crop out just the bounded object from the frame
        cropped = frame[cy - cr:cy + cr, cx - cr:cx + cr]

        # Now, insert the cropped object into the frame with random noise
        blank_image[y:y + cropped.shape[0], x:x + cropped.shape[1]] = cropped

        if frame_write is None:
            frame_write = blank_image.copy()

        # Now, send the entire frame back into
        # our object recognition model for confirmation
        center, confidence, (x, y), radius, frame_write, bbox = \
            check_for_initial_target(blank_image,
                                     img_write=frame_write, show_img=show_img,
                                     in_debug=in_debug)

        if show_img:
            cv2.imshow("cropped", frame_write)

        # Return whether or not the object was confirmed.
        if confidence is not None \
                and confidence > .2:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def check_for_initial_target(img=None, img_write=None, show_img=False, in_debug=False):
    if img is None:
        img = get_cur_frame()

    if img_write is None:
        img_write = img.copy()

    my_color = (20, 20, 230)
    b_boxes, scores, class_ids = detect_object(img, img_write, net=visdrone_net)

    scores_kept = []
    b_boxes_kept = []

    # only consider people here...
    for index in range(0, len(scores)):

        if not in_debug:
            if (visdrone_classes[class_ids[index]] == "pedestrian"
                    or visdrone_classes[class_ids[index]] == "people"):
                scores_kept.append(scores[index])
                b_boxes_kept.append(b_boxes[index])
        else:
            if visdrone_classes[class_ids[index]] == "car":
                scores_kept.append(scores[index])
                b_boxes_kept.append(b_boxes[index])

    if len(scores_kept) >= 1:

        # Take the bounded object with the maximum confidence score
        max_confidence = max(scores_kept)
        max_index = scores_kept.index(max_confidence)

        x, y, w, h = b_boxes_kept[max_index]
        cv2.rectangle(img_write, (x, y), (x + w, y + h), (20, 20, 230), 2)
        cv2.putText(img_write, f"{visdrone_classes[class_ids[max_index]]}, {scores_kept[max_index]}"
                    , (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, my_color, 2)

        if show_img:
            cv2.imshow("Real-time Detect", img_write)

        # Calculate where the center of the object is in the frame
        x_center = x + int(w / 2)
        y_center = y + (int(h / 2))

        return (x_center, y_center), max_confidence, (x, y), max([h / 2, w / 2]), img_write, b_boxes_kept[max_index]
    else:
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)


def track_with_confirm(img, img_write=None, show_img=False, debug_mode=False):
    global total_track_misses, confirmed_object_tracking

    if img_write is None:
        img_write = img.copy()

    # Here, we will use an object track to ensure we're
    #       tracking the very same object we identified earlier.
    center, confidence, (x, y), radius, frm_display, bbox = \
        track_object(img, show_img=show_img)

    if confidence is not None:

        if confirm_obj_in_bbox(img, frame_write=None,
                               bbox=bbox,
                               show_img=show_img,
                               in_debug=debug_mode):
            total_track_misses = 0
            confirmed_object_tracking = True
        else:
            total_track_misses += 1
    else:
        # Tracking failure
        total_track_misses += 1

    if total_track_misses >= TRACKER_MISSES_MAX:
        # Tracking failure
        confirmed_object_tracking = False
        confidence = None
    else:
        cv2.putText(frm_display, "Tracking...", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2)

    if show_img:
        cv2.imshow("Real-time Detect", img_write)

    return center, confidence, (x, y), radius, frm_display, bbox


def track_object(img, img_write=None, show_img=False, debug=False):
    # Here, we will use an object tracker to ensure we're
    #       tracking the very same object we identified earlier.

    if img_write is None:
        img_write = img.copy()

    ok, box = tracker.update(img)

    if ok:
        bbox = tuple(int(val) for val in box)
        x, y, w, h = bbox
        x_center = bbox[0] + int(bbox[2] / 2)
        y_center = bbox[1] + (int(bbox[3] / 2))
        cv2.rectangle(img_write, (x, y), (x + w, y + h), (255, 0, 255), 2)

        if show_img:
            cv2.imshow("Real-time Detect", img_write)

        return (x_center, y_center), 1.0, (x, y), max([h / 2, w / 2]), img_write, bbox
    else:
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)


def detect_object(img, img_write, net, show_img=False, debug=False):
    if img_write is None:
        img_write = img.copy()

    cv2.putText(img_write, 'detecting...', (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (192, 192), swapRB=True, crop=False)

    # blob = cv2.dnn.blobFromImage(
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
                    (detection[0:4] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # hold our final results here...
    final_bboxes = []
    final_scores = []
    final_class_ids = []

    if len(b_boxes) > 0:
        # Perform non maximum suppression for the bounding boxes
        # to filter overlapping and low confidence bounding boxes.
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten()
        for index in indices:
            final_bboxes.append(b_boxes[index])
            final_class_ids.append(class_ids[index])
            final_scores.append(confidences[index])

    return final_bboxes, final_scores, final_class_ids


def start_camera_stream():
    # comment our when not testing in sim...

    global pipeline, config, \
        FRAME_HORIZONTAL_CENTER, \
        FRAME_VERTICAL_CENTER, \
        FRAME_HEIGHT, FRAME_WIDTH

    pipeline = rs.pipeline()
    config = rs.config()

    # Keep in mind that we are using BGR ordering because
    # it works natively with openCV..
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    image_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    image_intrinsics = image_profile.get_intrinsics()
    frame_w, frame_h = image_intrinsics.width, image_intrinsics.height

    FRAME_HEIGHT = int(frame_h)
    FRAME_WIDTH = int(frame_w)
    FRAME_HORIZONTAL_CENTER = int(FRAME_WIDTH/2)
    FRAME_VERTICAL_CENTER = int(FRAME_HEIGHT/2)


def get_cur_frame(attempts=5, flip_v=False):
    # Wait for a coherent pair of frames: depth and color
    tries = 0

    # This will capture the frames from the simulator.
    # If using an actual camera, comment out the two lines of
    # code below and replace with code that returns a single frame
    # from your camera.
    # image = fg_camera_sim.get_cur_frame()
    # return cv2.resize(image, (int(FRAME_HORIZONTAL_CENTER * 2), int(FRAME_VERTICAL_CENTER * 2)))

    # Code below can be used with the realsense camera...
    while tries <= attempts:
        try:
            frames = pipeline.wait_for_frames()
            rgb_frame = frames.get_color_frame()
            rgb_frame = np.asanyarray(rgb_frame.get_data())

            if flip_v:
                rgb_frame = cv2.flip(rgb_frame, 0)
            return rgb_frame
        except Exception as e:
            print(e)

        tries += 1


def set_object_to_track(frame, bbox, bbox_margin=40):
    # On some platforms, the tracker reset doesn't work,
    # so we need to create a new instance here.
    create_tracker(DEFAULT_TRACKER_TYPE)

    if bbox_margin <= 0:
        tracker.init(frame, bbox)
    else:
        # center original bbox within a larger, square bbox
        x, y, w, h = bbox

        # get the center and the radius
        cx = x + w // 2
        cy = y + h // 2
        cr = max(w, h) // 2

        r = cr + bbox_margin
        new_bbox = [cx - r, cy - r, r * 2, r * 2]
        x, y, w, h = new_bbox
        tracker.init(frame, (x, y, w, h))


if __name__ == '__main__':

    # init video
    start_camera_stream()
    load_visdrone_network()
    object_identified = False

    while True:
        # Start timer
        timer = cv2.getTickCount()
        frame = get_cur_frame()
        frm_display = frame.copy()

        if not object_identified:
            center, confidence, (x, y), radius, frm_display, bbox \
                = check_for_initial_target(frame, frm_display,
                                           in_debug=False, show_img=True)
            if confidence is not None \
                    and confidence > .2:
                # Initialize tracker with first frame and bounding box
                # bbox needs: xb,yb,wb,hb
                object_identified = True
                set_object_to_track(frame, bbox)
        else:

            center, confidence, (x, y), radius, frm_display, bbox \
                = track_with_confirm(frame, frm_display,
                                     debug_mode=False, show_img=True)

            if not confidence:
                cv2.putText(frm_display,
                            "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)

                object_identified = False

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frm_display, "FPS : " + str(int(fps)),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Real-time Detect", frm_display)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

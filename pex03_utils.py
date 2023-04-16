
import glob
import logging
import os
import shutil
import time
from pathlib import Path
import cv2

GRIPPER_OPEN = 1087
GRIPPER_CLOSED = 1940

# TODO: be sure to set this path to your team's name (instead of "cam")
#   - set to something that works on your text machine.
DEFAULT_LOG_PATH = '/media/usafa/data/pex02_mission/cam'


def write_log_entry(entry):
    logging.info(entry)


def write_frame(frm_num, frame, path):
    frm = "{:06d}".format(int(frm_num))
    cv2.imwrite(f"{path}/frm_{frm}.png", frame)


def get_ground_distance(height, hypotenuse):
    import math

    # Assuming we know the distance to object from the air
    # (the hypotenuse), we can calculate the ground distance
    # by using the simple formula of:
    # d^2 = hypotenuse^2 - height^2

    return math.sqrt(hypotenuse ** 2 - height ** 2)


def calc_new_location(cur_lat, cur_lon, heading, meters):
    from geopy import distance
    from geopy import Point

    # given: cur_lat, cur_lon,
    #        heading = bearing in degrees,
    #        meters = distance in meters

    origin = Point(cur_lat, cur_lon)
    destination = distance.distance(
        kilometers=(meters * .001)).destination(origin, heading)

    return destination.latitude, destination.longitude


def get_avg_distance_to_obj(seconds, device, virtual_mode=False):
    if virtual_mode:
        return 35.0

    distance = device.rangefinder.distance
    i = 1

    if distance is None:
        return -1

    t_end = time.time() + seconds
    while time.time() < t_end:
        i += 1
        distance += device.rangefinder.distance

    return distance / i


def release_grip(drone, seconds=2):
    sec = 1

    while sec <= seconds:
        override_gripper_state(drone, state=GRIPPER_OPEN)
        time.sleep(1)
        sec += 1


def override_gripper_state(drone, state=GRIPPER_CLOSED, channel=7):
    drone.channels.overrides[f'{channel}'] = state


def backup_prev_experiment(path):
    if os.path.exists(path):
        if len(glob.glob(f'{path}/*')) > 0:
            time_stamp = time.time()
            shutil.move(os.path.normpath(path),
                        os.path.normpath(f'{path}_{time_stamp}'))

    Path(path).mkdir(parents=True, exist_ok=True)


def clear_path(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)

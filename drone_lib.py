import math
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import traceback
import sys
import logging

MAV_FRAME_BODY_FRD = 12


# Some useful information on coordinates and reference frames can be found
# here: https://dronekit-python.readthedocs.io/en/latest/guide/copter/guided_mode.html#guided-mode-copter-velocity-control

def display_rover_state(connection):
    print(f"Battery: {connection.battery}")
    print(f"Last heartbeat: {connection.last_heartbeat}")
    print(f"Attitude: {connection.attitude}")
    print(f"Heading: {connection.heading}")
    print(f"Ground speed:{connection.groundspeed}")
    print(f"Velocity: {connection.velocity}")
    print(f"Steering rc: {connection.channels[1]}")
    print(f"Throttle rc: {connection.channels[3]}")


def display_vehicle_state(connection):
    # vehicle is an instance of the Vehicle class
    print(f"Autopilot capabilities (supports ftp): {connection.capabilities.ftp}")
    print(f"Global location: {connection.location.global_frame}")
    print(f"Global location (relative altitude): {connection.location.global_relative_frame}")
    print(f"Local location: {connection.location.local_frame}")  # NED
    print(f"Attitude: {connection.attitude}")
    print(f"Velocity: {connection.velocity}")
    print(f"GPS: {connection.gps_0}")
    print(f"Ground speed:{connection.groundspeed}")
    print(f"Airspeed: {connection.airspeed}")
    print(f"Gimbal status: {connection.gimbal}")
    print(f"Battery: {connection.battery}")
    print(f"EKF OK?: {connection.ekf_ok}")
    print(f"Last heartbeat: {connection.last_heartbeat}")
    print(f"Rangefinder: {connection.rangefinder}")
    print(f"Rangefinder distance: {connection.rangefinder.distance}")
    print(f"Rangefinder voltage: {connection.rangefinder.voltage}")
    print(f"Heading: {connection.heading}")
    print(f"Is Armable?: {connection.is_armable}")
    print(f"System status: {connection.system_status.state}")
    print(f"Mode: {connection.mode.name}")  # settable
    print(f"Armed: {connection.armed}")  # settable


def get_short_distance_meters(location_1, location_2):
    """
    Returns the ground distance in meters between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    d_lat = location_2.lat - location_1.lat
    d_long = location_2.lon - location_1.lon
    return math.sqrt((d_lat * d_lat) + (d_long * d_long)) * 1.113195e5


def device_relative_distance_from_point(device, lat, lon, alt):
    """
    Gets distance in meters to the current waypoint.

    """

    cur_location = device.location.global_relative_frame
    target_location = LocationGlobalRelative(lat, lon, alt)
    distance = get_short_distance_meters(cur_location, target_location)

    return distance


def log_activity(msg, log=None):
    if log is None:
        logging.info(msg)
    else:
        log.info(msg)

    print(msg)


def small_move_up(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone upwards by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """

    # Note: up is negative
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, 0, 0, -velocity, duration)
    move_local(device, 0, 0, velocity, duration, log, cube=cube)


def small_move_down(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone upwards by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """

    # Note: down is positive
    velocity = abs(velocity)

    # send_body_frame_velocities(device, 0, 0, velocity, duration)
    move_local(device, 0, 0, velocity, duration, log, cube=cube)


def small_move_forward(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone forward by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = abs(velocity)

    # send_body_frame_velocities(device, velocity, 0, 0.0, duration)
    move_local(device, velocity, 0, 0.0, duration, log, cube=cube)


def small_move_back(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone backward by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, velocity, 0, 0.0, duration)
    move_local(device, velocity, 0, 0.0, duration, log, cube=cube)


def small_move_right(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone right by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = abs(velocity)

    # send_body_frame_velocities(device, 0, velocity, 0.0, duration)
    move_local(device, 0, velocity, 0.0, duration, log, cube=cube)


def small_move_left(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone left by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, 0, velocity, 0.0, duration)
    move_local(device, 0, velocity, 0.0, duration, log, cube=cube)


def move_local(device, x, y, z, duration=1, log=None, cube=True):
    log_activity(f"Local move with velocities {x},{y},{z} for {duration} seconds.", log)
    send_body_frame_velocities(device, x, y, z, duration)
    return

    '''if cube:
        send_body_frame_velocities(device, x, y, z, duration)
    else:
        send_global_frame_velocities(device, -y, -x, z, duration)'''


def condition_yaw(device, heading, relative=False, log=None):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.

    By default the yaw of the vehicle will follow the direction of travel. After setting
    the yaw using this function there is no way to return to the default yaw "follow direction
    of travel" behavior (https://github.com/diydrones/ardupilot/issues/2427)

    For more information see:
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
    """

    log_activity(f"Yaw to {heading} degrees (relative = {relative}).", log)

    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle

    # create the CONDITION_YAW command using command_long_encode()
    msg = device.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        heading,  # param 1, yaw in degrees
        0,  # param 2, yaw speed deg/s
        1,  # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)  # param 5 ~ 7 not used

    # send command to vehicle
    device.send_mavlink(msg)


def send_global_frame_velocities(device, velocity_x, velocity_y, velocity_z, duration=2):
    # To move up, down, left, right relative to the drone's current position,
    # you need to create a vehicle.message_factory.set_position_target_local_ned_encode.
    # It will require a frame of mavutil.mavlink.MAV_FRAME_BODY_NED
    # (north, east, down reference).
    # You then add the required x,y and/or z velocities (in m/s) to the message.

    msg = device.message_factory.set_position_target_global_int_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0,  # lat_int - X Position in WGS84 frame in 1e7 * meters
        0,  # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0,  # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        velocity_x,  # X velocity in NED frame in m/s
        velocity_y,  # Y velocity in NED frame in m/s
        velocity_z,  # Z velocity in NED frame in m/s
        0, 0, 0,  # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        device.send_mavlink(msg)
        time.sleep(1)


def send_body_frame_velocities(device, forward, right, velocity_z, duration=2):
    # To move up, down, left, right, forward, back independent of North or East directions,
    # you need to create a vehicle.message_factory.set_position_target_local_ned_encode.
    # It will require a frame of mavutil.mavlink.MAV_FRAME_BODY_NED.
    # You then add the required x,y and/or z velocities (in m/s) to the message.

    # NOTE: according to documentation (hard to get), MAV_FRAME_BODY_NED
    #   is deprecated. We "SHOULD" be using MAV_FRAME_BODY_FRD
    # SEE: https://mavlink.io/en/messages/common.html#MAV_FRAME_BODY_NED

    # Body fixed frame of reference, Z-down (x: Forward, y: Right, z: Down).

    # NOTE: The velocity_z component is perpendicular to the plane of
    # velocity_x and velocity_y, with a positive value towards the ground,
    # following the right-hand convention.

    msg = device.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0,  # lat_int - X Position in WGS84 frame in 1e7 * meters
        0,  # lon_int - Y Position in WGS84 frame in 1e7 * meters
        0,  # alt - Altitude in meters in AMSL altitude(not WGS84 if absolute or relative)
        # altitude above terrain if GLOBAL_TERRAIN_ALT_INT
        forward,  # X velocity in body frame in m/s
        right,  # Y velocity in body frame in m/s
        velocity_z,  # Z velocity in body frame in m/s
        0, 0, 0,  # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        device.send_mavlink(msg)
        time.sleep(1)


def connect_device(s_connection, baud=115200, log=None):
    log_activity(f"Connecting to device {s_connection} with baud rate {baud}...", log)
    device = connect(s_connection, wait_ready=True, baud=baud)
    log_activity("Device connected.", log)
    log_activity(f"Device version: {device.version}, log=")
    return device


def arm_device(device, log=None, n_reps=10, mode="GUIDED"):
    log_activity("Arming device...", log)
    wait = 1

    # "GUIDED" mode sets drone to listen
    # for our commands that tell it what to do...
    device.mode = VehicleMode(mode)
    while device.mode != mode:
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            break

        log_activity(f"Switching to {mode} mode...", log)
        time.sleep(2)
        wait += 1
    wait = 1
    device.armed = True

    while not device.armed:
        if wait > n_reps:
            log_activity("arm timeout.", log)
            break
        log_activity("Waiting for arm...", log)
        time.sleep(2)

    log_activity(f"Device armed: {device.armed}.", log)

    return device.armed


def change_device_mode(device, mode, n_reps=10, log=None):
    wait = 0
    log_activity(f"Changing device mode from {device.mode} to {mode}...", log)

    device.mode = VehicleMode(mode)

    while device.mode != mode:
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            return False
        device.mode = VehicleMode(mode)
        time.sleep(.5)
        wait += 1

    log_activity(f"Device mode = {device.mode}.", log)


def device_takeoff(device, altitude, log=None):
    log_activity("Device takeoff...", log)
    device.mode = VehicleMode("GUIDED")
    time.sleep(.5)
    device.simple_takeoff(altitude)
    device.airspeed = 3
    while device.armed \
            and device.mode == "GUIDED":
        log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
        if device.location.global_relative_frame.alt >= (altitude * .90):
            break
        time.sleep(.5)

    time.sleep(2)


def device_land(device, log=None):
    log_activity("Device land...", log)
    device.mode = VehicleMode("LAND")
    time.sleep(.5)
    while device.armed \
            and device.mode == "LAND":
        log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
        if device.location.global_relative_frame.alt <= 1:
            log_activity("Device has landed.", log)
            break
        time.sleep(.1)

    # disarm the device
    time.sleep(4)
    device.armed = False


def execute_flight_plan(device, n_reps=10, wait=1, log=None):
    if device.commands.count == 0:
        log_activity("No flight plan to execute.", log)
        return False

    log_activity("Executing flight plan...", log)

    # Reset mission set to first (0) waypoint
    device.commands.next = 0

    # Set mode to AUTO to start mission
    device.mode = VehicleMode("AUTO")
    time.sleep(.5)
    while device.mode != "AUTO":
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            return False

        log_activity("Switching to AUTO mode...", log)
        time.sleep(1)
        wait += 1
    return True


def goto_point(device, lat, lon, speed, alt, log=None):
    log_activity(f"Goto point: {lat}, {lon}, {speed}, {alt}...", log)

    # set the default travel speed
    device.airspeed = speed

    point = LocationGlobalRelative(lat, lon, alt)

    device.simple_goto(point)

    while device.armed \
            and device.mode == "GUIDED":
        try:
            log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
            log_activity(f"Current lat: {device.location.global_relative_frame.lat}", log)
            log_activity(f"Current lon: {device.location.global_relative_frame.lon, log}")

            alt_percent = device.location.global_relative_frame.alt / alt
            lat_percent = device.location.global_relative_frame.lat / lat
            lon_percent = device.location.global_relative_frame.lon / lon

            log_activity(f"Relative position to destination: {alt_percent},{lat_percent}, {lon_percent}", log)

            if (0.99 <= alt_percent <= 1.1) \
                    and (.99 <= lat_percent <= 1.1) \
                    and (.99 <= lon_percent <= 1.1):
                break  # close enough - may never be perfectly on the mark
            time.sleep(1)
        except Exception as e:
            log_activity(f"Error on goto: {traceback.format_exception(*sys.exc_info())}", log)
            raise


def goto_point2(device, lat, lon, speed, alt, log=None, wait_secs=None):
    log_activity(f"Goto point: {lat}, {lon}, {speed}, {alt}...", log)

    # set the default travel speed
    device.airspeed = speed

    point = LocationGlobalRelative(lat, lon, alt)

    device.simple_goto(point)

    while device.armed \
            and device.mode == "GUIDED":
        try:
            cur_alt = device.location.global_relative_frame.alt
            log_activity(f"Current altitude: {cur_alt}", log)
            log_activity(f"Current lat: {device.location.global_relative_frame.lat}", log)
            log_activity(f"Current lon: {device.location.global_relative_frame.lon, log}")

            if cur_alt <= alt:
                alt_percent = cur_alt / alt
            else:
                alt_percent = alt / cur_alt

            distance = device_relative_distance_from_point(device, lat, lon, alt)

            log_activity(f"Ground distance to destination: {distance}; diff in altitude: {alt_percent} ", log)

            if wait_secs is None:
                # Monitor our progress...
                if (.985 <= alt_percent <= 1.1) \
                        and distance < 1.2:  # (less than 1.2 meters in distance)
                    break  # close enough - may never be perfectly on the mark
            else:
                # Instead of monitoring progress towards destination,
                # just wait the allotted time and then get out.
                time.sleep(wait_secs)
                break

            time.sleep(1.0)
        except Exception as e:
            log_activity(f"Error on goto: {traceback.format_exception(*sys.exc_info())}", log)
            raise


def return_to_launch(device, log=None):
    log_activity("Device returning to launch...", log)
    device.mode = VehicleMode("RTL")
    time.sleep(.5)
    while device.armed \
            and device.mode == "RTL":
        log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
        if device.location.global_relative_frame.alt <= .01:
            log_activity("Device has landed.", log)
            break
        time.sleep(.5)

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import sys
import socket


def connect_device(s_connection):
    print("Connecting to device...")
    device = connect(ip=s_connection, wait_ready=True)
    print("Device connected.")
    print(f"Device version: {device.version}")
    return device


def arm_device(device):
    while not device.is_armable:
        print("Switching device to armable...")
        time.sleep(2)
        # "GUIDED" mode sets drone to listen
        # for our commands that tell it what to do...
    while device.mode != "GUIDED":
        print("Switching to GUIDED mode...")
        device.mode = VehicleMode("GUIDED")
        time.sleep(2)
    while not device.armed:
        print("Waiting for arm...")
        time.sleep(2)
        device.armed = True


def device_takeoff(device, altitude):
    print("Device takeoff...")
    device.mode = VehicleMode("GUIDED")
    device.simple_takeoff(altitude)
    device.airspeed = 3
    while True:
        print(f"Current altitude: {device.location.global_relative_frame.alt}")
        if device.location.global_relative_frame.alt >= (altitude * .95):
            break
        time.sleep(.5)


def device_land(device):
    print("Device land...")
    device.mode = VehicleMode("LAND")
    while device.armed:
        print(f"Current altitude: {device.location.global_relative_frame.alt}")
        time.sleep(.1)
        print("Device has landed.")


def goto_point(device, lat, lon, speed, alt):
    print(f"Goto point: {lat}, {lon}, {speed}, {alt}...")
    point = LocationGlobalRelative(lat, lon, alt)
    device.simple_goto(point)
    # set the default travel speed
    device.airspeed = speed
    while (device.location.global_relative_frame.alt != alt
           and device.location.global_relative_frame.lon != lon
           and device.location.global_relative_frame.lat != lat):
        print(f"Current altitude: {device.location.global_relative_frame.alt}")
        print(f"Current lat: {device.location.global_relative_frame.lat}")
        print(f"Current lon: {device.location.global_relative_frame.lon}")


def return_to_launch(device):
    print("Device returning to launch...")
    device.mode = VehicleMode("RTL")
    while device.armed:
        print(f"Current altitude: {device.location.global_relative_frame.alt}")
        time.sleep(.5)


def main(args):
    print("Device has landed.")
    # *** Test Flight ***
    # First, connect to the autopilot
    drone = connect_device("127.0.0.1:14550")
    # Next, arm the device...
    arm_device(drone)
    # takeoff and climb 45 meters
    device_takeoff(device=drone, altitude=45)
    # Fly to the following lat/long points...
    goto_point(device=drone, lat=21.326268,
               lon=-157.932869, speed=50, alt=100)
    goto_point(device=drone, lat=21.329858,
               lon=-157.940820, speed=50, alt=100)
    goto_point(device=drone, lat=21.328642,
               lon=-157.952017, speed=50, alt=100)
    goto_point(device=drone, lat=21.317618,
               lon=-157.934842, speed=50, alt=100)
    # Fly back to home location...
    goto_point(drone, 21.326268, -157.932869, 50, 100)
    # Return to home and land
    return_to_launch(device=drone)
    # Disconnect from autopilot
    drone.close()
    print("End of demonstration.")


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)

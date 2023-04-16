
import logging
import time
import cv2
import random
import sys
import traceback
from student import drone_lib
from student import object_tracking as obj_track
from student import pex03_utils

# Various mission states:
# We start out in "seek" mode, if we think we have a target, we move to "confirm" mode,
# If target not confirmed, we move back to "seek" mode.
# Once a target is confirmed, we move to "target" mode.
# After positioning to target and calculating a drop point, we move to "deliver" mode
# After delivering package, we move to RTL to return home.
MISSION_MODE_SEEK = 0
MISSION_MODE_CONFIRM = 1
MISSION_MODE_TARGET = 2
MISSION_MODE_DELIVER = 4
MISSION_MODE_RTL = 8

DEFAULT_UPDATE_RATE = 1  # How many frames do we wait to execute on.
DEFAULT_TARGET_RADIUS_MULTI = 1.0  # 1.2 x the radius of the target is good for this mission.
DEFAULT_TARGET_RADIUS = 5
DEFAULT_IMG_WRITE_RATE = 4  # write every N frames to disk...
DEFAULT_MAX_CONFIRM_ATTEMPTS = 8

# Font for use with the information window
IMG_FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_SNAPSHOT_PATH = '/media/usafa/data/pex03_mission/cam_pex'
MISSION_VIRTUAL_MODE = True


class DroneMission:

    def __init__(self, device,
                 virtual_mode=True,
                 update_rate=DEFAULT_UPDATE_RATE,
                 target_radius=DEFAULT_TARGET_RADIUS,
                 target_multiplier=DEFAULT_TARGET_RADIUS_MULTI,
                 image_log_rate=DEFAULT_IMG_WRITE_RATE,
                 log_write_path=pex03_utils.DEFAULT_LOG_PATH,
                 max_confirm_attempts=DEFAULT_MAX_CONFIRM_ATTEMPTS,
                 min_target_radius=10,
                 mission_start_mode=MISSION_MODE_SEEK):

        self.drone = device
        self.log = None  # logger instance
        self.virtual_mode = virtual_mode
        self.update_rate = update_rate
        self.target_radius = target_radius
        self.target_multiplier = target_multiplier
        self.image_log_rate = image_log_rate
        self.log_path = log_write_path
        self.max_confirm_attempts = max_confirm_attempts
        self.target_radius = min_target_radius

        # Mission tracking variables
        self.mission_mode = mission_start_mode
        self.target_locate_attempts = 0
        self.refresh_counter = 0
        self.confirm_attempts = 0
        self.direction_x = "unknown"
        self.direction_y = "unknown"
        self.inside_circle = False
        self.object_identified = False
        self.target_locate_attempts = 0

        # Info related to last (potential) target sighting
        self.init_obj_lon = None
        self.init_obj_lat = None
        self.init_obj_alt = None
        self.init_obj_heading = None
        self.init_obj_point = None  # center point in pixels

        # Holds last drone position snapshot
        self.last_lon_pos = -1.0
        self.last_lat_pos = -1.0
        self.last_alt_pos = -1.0
        self.last_heading_pos = -1.0

    @staticmethod
    def log_info(msg):
        log.info(msg)

    def arm_drone(self):

        drone_lib.arm_device(self.drone)

    def target_is_centered(self, target_point, frame_write=None):

        # Determine how far off center from target we are...
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])

        self.log_info(f"Current alignment with target center: dx={dx}, dy={dy}")

        x, y = target_point

        # Draw a line between most-recent center point of target and
        # the drone's position (i.e. the center of the frame).
        if frame_write is not None:
            cv2.line(frame_write, target_point,
                     (int(obj_track.FRAME_HORIZONTAL_CENTER),
                      int(obj_track.FRAME_VERTICAL_CENTER)),
                     (0, 0, 255), 5)

            cv2.circle(frame_write, (int(x), int(y)), int(self.target_radius), (0, 0, 255), 2)
            cv2.circle(frame_write, (int(x), int(y)), int(self.target_radius * self.target_multiplier),
                       (255, 255, 0), 2)

            cv2.circle(frame_write, target_point, 5, (0, 255, 255), -1)

        return self.check_in_circle(target_point)

    def check_in_circle(self, target_point):
        # Check to see if we're inside our safe zone relative to target...
        if (int(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER) ** 2 \
                + (int(target_point[1]) - obj_track.FRAME_VERTICAL_CENTER) ** 2 \
                <= self.target_radius ** 2:

            return True
        else:
            return False

    def switch_mission_to_confirm_mode(self):

        # Time to pause and take a closer look at the target...
        # You can stop/pause the current flight plan by switching out of AUTO mode (e.g. into GUIDED mode).
        # If you switch back to AUTO mode the mission will either restart at the beginning or
        # resume at the current waypoint.
        # The behaviour depends on the value of the MIS_RESTART parameter.
        # (We want to be sure that the drone is configured with MIS_RESTART = 0)
        self.mission_mode = MISSION_MODE_CONFIRM  # we will confirm the target next time this function is called.
        self.confirm_attempts = 0
        self.log_info(f"Need to confirm target.")

        # Drone's internal mission is temporarily
        # suspended until we can confirm target.
        self.log_info(f"Switching drone to GUIDED...")
        drone_lib.change_device_mode(device=self.drone, mode="GUIDED")

        # First, re-position over the point where
        # where the target was first spotted, and
        # increase altitude by 5 meters to better fit the
        # target in the incoming images.
        self.log_info(f"Positioning towards potential target...")
        drone_lib.goto_point(self.drone,
                             self.init_obj_lat,
                             self.init_obj_lon,
                             2.5,
                             self.init_obj_alt)

        drone_lib.condition_yaw(self.drone, self.last_heading_pos)
        time.sleep(4)

    def confirm_objective(self, frame_write=None):

        if self.mission_mode == MISSION_MODE_CONFIRM:

            # If we still have a lock on the objective,
            # then switch over to targeting the objective...
            if self.object_identified:
                self.log_info(f"Target CONFIRMED.")

                # Begin the landing process by switching over to
                # "target" mode; while in this mode we attempt
                # to CENTER the target before taking distance measurement and
                # ultimately delivering the package to the objective.
                self.mission_mode = MISSION_MODE_TARGET
                return True

            else:

                if self.confirm_attempts \
                        >= self.max_confirm_attempts:

                    # We lost our objective... switch back to seek mode to
                    # continue looking for it.
                    self.mission_mode = MISSION_MODE_SEEK
                    self.log_info(f"Exceeded number of attempts.")
                    self.log_info(f"Switching back to AUTO...")
                    drone_lib.change_device_mode(device=self.drone, mode="AUTO")
                    return False

                else:
                    # We must've lost the target... try to get it back.
                    self.log_info("Re-acquiring target...")

                    if frame_write is not None:
                        cv2.putText(frame_write, "Re-acquiring target...",
                                    (10, 250), IMG_FONT, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)

                    # Move to point of original sighting.
                    drone_lib.goto_point(self.drone,
                                         self.init_obj_lat,
                                         self.init_obj_lon,
                                         2.5,
                                         self.init_obj_alt)

                    # Now, perform a random yaw for a
                    # different vantage point than before.
                    rand_number = random.random()
                    degrees = rand_number * 180
                    drone_lib.condition_yaw(self.drone, degrees)
                    time.sleep(2)
                    self.confirm_attempts += 1
        else:
            # NOTE: We might want to throw exception here,
            # since we are not in the correct mission mode to perform this check.
            return False

    def adjust_to_target_center(self, target_point, frame_write=None):

        # TODO: Calculate how far from center the target point (i.e. our person) is.
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])

        pixel_forgiveness = 1  # TODO: you decide what "good enough" is to consider centered on x or y axis...

        if self.mission_mode == MISSION_MODE_TARGET:

            # Adjust position relative to target
            if target_point is not None \
                    and self.object_identified:

                self.log_info("Targeting object...")

                if dx < 0:
                    self.direction_x = "L"
                if dx > 0:
                    self.direction_x = "R"
                if dy < 0:
                    self.direction_y = "B"
                if dy > 0:
                    self.direction_y = "F"
                if abs(dx) <= pixel_forgiveness:  # if we are within N pixels, no need to make adjustment
                    self.direction_x = "C"
                if abs(dy) <= pixel_forgiveness:  # if we are within N pixels, no need to make adjustment
                    self.direction_y = "C"

                self.log_info(f"target diff: dy={dy}, dx={dx}.")

                if (self.direction_y == "C"
                    and self.direction_x == "C") \
                        or self.inside_circle:

                    # time to deliver...
                    # Switch to "deliver" mode now.
                    self.mission_mode = MISSION_MODE_DELIVER
                    drone_lib.log_activity("Time to deliver package!")

                    if frame_write is not None:
                        cv2.putText(frame_write, "Deliver package!...",
                                    (10, 400), IMG_FONT, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    # TODO: see below for setting your threshold... you decide what it should be
                    pixel_distance_threshold = -1

                    # log movements...
                    logging.info("Targeting... determined changes in velocities: X: "
                                 + self.direction_x + ", Y: "
                                 + self.direction_y)

                    if frame_write is not None:
                        cv2.putText(frame_write, "Targeting...",
                                    (10, 400), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    if abs(dx) > pixel_distance_threshold:
                        # TODO: calculate a velocity for x-axis adjustment
                        # xv = ???
                        pass  # TODO: remove "pass" when you've completed this condition.
                    else:
                        # xv = ???
                        pass  # TODO: remove "pass" when you've completed this condition.
                    if abs(dy) > pixel_distance_threshold:
                        # yv =???
                        pass  # TODO: remove "pass" when you've completed this condition.
                    else:
                        # yv = ???
                        pass  # TODO: remove "pass" when you've completed this condition.

                    # Execute movements towards centering on target
                    if self.direction_y != "C":  # If we are not centered on y-axis...
                        if self.direction_y == "F":

                            if frame_write is not None:
                                cv2.putText(frame_write, "Move forward....", (10, 200), IMG_FONT, 1,
                                            (0, 255, 0), 2, cv2.LINE_AA)
                                # TODO: you can perform "drone_lib.small_move_forward here,
                                #       or you can do a drone_lib.move_local as well.
                                pass  # TODO: remove "pass" when you've completed this condition.
                        else:
                            if frame_write is not None:
                                cv2.putText(frame_write, "Move back....", (10, 200), IMG_FONT, 1,
                                            (0, 255, 0), 2, cv2.LINE_AA)

                            # TODO: you can perform "drone_lib.small_move_back here,
                            #       or you can do a drone_lib.move_local as well.
                            pass  # TODO: remove "pass" when you've completed this condition.

                    if self.direction_x != "C":  # If we are not centered on x-axis...

                        if self.direction_x == "R":

                            if frame_write is not None:
                                cv2.putText(frame_write, "Move right....", (10, 300), IMG_FONT, 1,
                                            (0, 255, 0), 2, cv2.LINE_AA)

                            # TODO: you can perform "drone_lib.small_move_right here,
                            #       or you can do a drone_lib.move_local as well.
                            pass  # TODO: remove "pass" when you've completed this condition.
                        else:
                            if frame_write is not None:
                                cv2.putText(frame_write, "Move left....", (10, 300), IMG_FONT, 1,
                                            (0, 255, 0), 2, cv2.LINE_AA)

                            # TODO: you can perform "drone_lib.small_move_left here,
                            #       or you can do a drone_lib.move_local as well.
                            pass  # TODO: remove "pass" when you've completed this condition.

            else:
                if frame_write is not None:
                    cv2.putText(frame_write, "LOST TARGET!", (10, 400), IMG_FONT, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)

                self.log_info("Cannot target object; switching back to seek mode...")
                self.mission_mode = MISSION_MODE_SEEK

    def deliver_package(self, frame_write=None):
        self.log_info("Delivering package...")

        location = self.drone.location.global_relative_frame
        lon = location.lon
        lat = location.lat
        alt = location.alt
        heading = self.drone.heading

        # First, get hypotenuse (distance to object from the air):
        dist_to_object = pex03_utils.get_avg_distance_to_obj(5.0, self.drone, self.virtual_mode)

        self.log_info(f"Hypotenuse: {dist_to_object}.")
        self.log_info(f"Altitude: {alt}.")

        if dist_to_object > 0:

            self.log_info(f"Calculating ground distance to object: "
                          f"using alt {alt} and distance from air {dist_to_object}...")

            # now get ground distance to object...
            ground_dist = 0
            # TODO: use pex03_utils.get_ground_distance to get ground distance
            self.log_info(f"Ground distance: {ground_dist}.")

            # TODO: now, calculate new lat/long within 10 feet of objective
            new_lat = new_lon = 0.0
            # Hint: use new_lat, new_lon = pex03_utils.calc_new_location function to get it...
            #       Don't forget that you want to deliver within ten fee of the person (not much closer),
            #       and you don't want to deliver too far away from that distance...
            self.log_info(f"Current pos: {lat}, {lon}, {heading}.")
            self.log_info(f"New location: {new_lat,}, {new_lon}.")

            # TODO: Now, goto new location...
            # HINT: drone_lib.goto_point(self.drone,.....)

            if frame_write is not None:
                cv2.putText(frame_write, "Delivering...", (10, 400), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
                pex03_utils.write_frame(self.refresh_counter, frame_write, self.log_path)

            if self.virtual_mode:
                # if just testing in sim, just land.
                self.log_info("Time to land...")
                self.mission_mode = MISSION_MODE_RTL

                # if in "virtual" mode, we will just land in spot where we want to drop package.
                drone_lib.device_land(self.drone)
            else:
                # TODO: ***** Finally, lower the package to the ground *****

                alt_thresh = -1  # TODO: YOU set here... when above a certain alt, what rate do you want to descend?
                # TODO: figure out your speeds here
                self.log_info("Lowering package....")
                while self.drone.location.global_relative_frame.alt > alt_thresh:
                    if self.drone.mode == "RTL" \
                            or self.drone.mode == "LAND" \
                            or self.mission_mode == MISSION_MODE_RTL:
                        logging.info("RTL/LAND mode activated.  Mission ended.")
                        break

                    # TODO: you can perform "drone_lib.small_move_down here,
                    #       or you can do a drone_lib.move_local as well.
                    pass  # TODO: remove "pass" when you've completed this condition.

                # TODO: after reach an alt below your threshold, how quickly should you continue to lower the package?
                while self.drone.location.global_relative_frame.alt > 3.20:
                    if self.drone.mode == "RTL" \
                            or self.drone.mode == "LAND" \
                            or self.mission_mode == MISSION_MODE_RTL:
                        logging.info("RTL/LAND mode activated.  Mission ended.")
                        break

                    # TODO: you can perform "drone_lib.small_move_down here,
                    #       or you can do a drone_lib.move_local as well.
                    pass  # TODO: remove "pass" when you've completed this condition.

            # TODO: Now, release the package.
            # TODO: see pex03_utils.release_grip function to
            #  figure out how to open the latch to release the package
            self.log_info("Releasing package now....")
            time.sleep(2)
        else:
            self.log_info("Could not get distance to object.")

        # Finally, return home.
        drone_lib.log_activity("Time to return home.")
        self.mission_mode = MISSION_MODE_RTL

        # TODO: return the drone home, your job is done...
        cv2.putText(frame_write, "Returning home...", (10, 500), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def determine_action(self, target_point, frame_write=None):

        self.log_info(f"Determine action...")
        self.log_info(f"Drone mode: {self.drone.mode}")

        if self.drone.mode == "RTL" \
                or self.drone.mode == "LAND" \
                or self.mission_mode == MISSION_MODE_RTL:
            # Do not execute anything,
            # mission has either ended or is aborted.

            self.log_info("Nothing to do. Landing...")
            return

        # Based on the (potential) target and the current mode we're in,
        # determine what actions the drone should take (if any)

        # Determine if we need to assess a potential target further...
        if self.mission_mode == MISSION_MODE_SEEK:
            if self.object_identified:

                # If we have not officially confirmed a target, let's attempt to do so
                # by switching to "confirm" mode and moving to a position over the
                # potential objective/target; we need to confirm the target.
                self.log_info("Switching to CONFIRM mode...")
                self.switch_mission_to_confirm_mode()
            else:

                # Still looking for objective...
                self.log_info("Looking for objective...")

                if self.drone.mode != "AUTO":
                    self.log_info(f"Switching drone to AUTO...")
                    drone_lib.change_device_mode(device=self.drone, mode="AUTO")

        else:  # We have an object in sight.  Now what?

            if target_point is not None \
                    and self.object_identified:  # So, we already have an object in our sight...

                # Determine if we're within a position that is satisfactory with respect to target.
                self.inside_circle = self.target_is_centered(target_point, frame_write)
                self.log_info(f"Inside target zone: {self.inside_circle}.")
            else:
                self.log_info("No target (yet).")

            # Confirm objective and switch to target mode (if able to confirm)
            if self.mission_mode == MISSION_MODE_CONFIRM:
                self.confirm_objective(frame_write)

        if self.mission_mode == MISSION_MODE_TARGET:
            # Center objective in the camera's frame...
            self.adjust_to_target_center(target_point, frame_write)

        # now, calculate distance to object.
        if self.mission_mode == MISSION_MODE_DELIVER:
            self.deliver_package(frame_write)

        if self.virtual_mode:
            cv2.imshow("Real-time Detect", frame_write)
            key = cv2.waitKey(1) & 0xFF

    def conduct_mission(self):

        # Here, we will loop until we find a human target and deliver the care package,
        # or until the drone's flight plan completes (and we land).
        self.log_info("Mission started...")
        self.object_identified = False

        while self.drone.armed:

            if self.drone.mode == "RTL" \
                    or self.drone.mode == "LAND" \
                    or self.mission_mode == MISSION_MODE_RTL:
                self.log_info("RTL/LAND mode activated.  Mission ended.")

                # Mission is over, jump out of the loop

                break

            timer = cv2.getTickCount()

            # Grab current frame from camera
            # TODO: use obj_track to get current frame
            frame = None

            # Take a snapshot of drone's current location
            # that corresponds with the frame
            location = self.drone.location.global_relative_frame

            # TODO: keep track of self.last_lon_pos, self.last_lat_pos = location.lat,
            #  HINT: self.last_alt_pos = location.alt, etc...
            self.last_heading_pos = self.drone.heading

            # Prep information frame (the frame we will draw on)
            # for logging/debugging purposes
            frm_display = frame.copy()

            if not self.object_identified:

                # We're trying to get an initial sighting.
                center, confidence, corner, radius, frm_display, bbox \
                    = obj_track.check_for_initial_target(frame, frm_display,
                                                         self.virtual_mode, self.virtual_mode)

                # TODO: set your confidence level here...
                # HINT: needs to be smaller than 99%!
                conf_level = .99
                if confidence is not None \
                        and confidence > conf_level:
                    # We found something.  Now, send to tracker.
                    # Initialize tracker with first frame and bounding box
                    # bbox needs: xb,yb,wb,hb
                    self.object_identified = True

                    # TODO: start tracking your object!
                    #HINT: obj_track.set_object_to_track

                    # Now, hold onto location where we first began
                    # tracking the object...
                    self.init_obj_lat = location.lat
                    self.init_obj_lon = location.lon
                    self.init_obj_alt = location.alt
                    self.init_obj_heading = self.drone.heading

                    # Record image of what was "confirmed" here.
                    pex03_utils.write_frame(self.refresh_counter, frm_display, self.log_path)
            else:

                # TODO: Continue to track our objective.
                #HINT: center, confidence, corner, radius, frm_display, bbox = obj_track.track_with_confirm

                if not confidence:
                    cv2.putText(frm_display,
                                "Tracking failure detected",
                                (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)

                    # We're going to try to re-acquire our objective.
                    self.object_identified = False

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(frm_display, "FPS : " + str(int(fps)),
                        (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            if self.virtual_mode:
                cv2.imshow("Real-time Detect", frm_display)
                cv2.imshow("Raw", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Time to adjust drone's position?
            if (self.refresh_counter % self.update_rate) == 0 \
                    or self.mission_mode != MISSION_MODE_SEEK:
                # determine drone's next actions (if any)
                self.determine_action(center, frame_write=frm_display)

            # Now, write frame with stats for informational purposes only
            if (self.refresh_counter % self.image_log_rate) == 0:
                pex03_utils.write_frame(self.refresh_counter, frm_display, self.log_path)

            self.refresh_counter += 1


if __name__ == '__main__':

    # log.info("backing up old mission...")
    pex03_utils.backup_prev_experiment(IMG_SNAPSHOT_PATH)

    # Setup a log file for recording important activities during our session.
    log_file = time.strftime(IMG_SNAPSHOT_PATH + "/Cam_PEX03_%Y%m%d-%H%M%S") + ".log"

    # prepare log file...
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    log = logging.getLogger(__name__)

    log.info("PEX 03 test program start.")
    log.info("Starting camera feed...")
    obj_track.start_camera_stream()

    log.info("Loading visdrone model...")
    obj_track.load_visdrone_network()
    log.info("Model loaded.")
    object_identified = False

    # Connect to the autopilot
    if MISSION_VIRTUAL_MODE:
        drone = drone_lib.connect_device("127.0.0.1:14550")
    else:
        drone = drone_lib.connect_device("/dev/ttyACM0", baud=115200)

    # Test latch - ensure open/close.
    # pex03_utils.release_grip(2)

    if not MISSION_VIRTUAL_MODE:
        if drone.rangefinder.distance is None:
            log.info("Rangefinder not detected.  Mission end")
            exit(99)
        else:
            log.info(
                f"Rangefinder avg distance check (before takeoff):"
                f" {pex03_utils.get_avg_distance_to_obj(2, drone)}")

    # If the autopilot has no mission, terminate program
    drone.commands.download()
    time.sleep(1)

    log.info("Looking for mission to execute...")
    if drone.commands.count < 1:
        log.info("No mission to execute.")
        exit(99)

    # Arm the drone.
    drone_lib.arm_device(drone, log=log)

    # takeoff and climb 45 feet
    drone_lib.device_takeoff(drone, 15, log=log)

    try:
        # start mission
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        drone_mission = DroneMission(device=drone,
                                     virtual_mode=MISSION_VIRTUAL_MODE,
                                     log_write_path=IMG_SNAPSHOT_PATH)

        # Now, look for target...
        drone_mission.conduct_mission()

        # Mission is over; disarm and disconnect.
        log.info("Disarming device...")
        drone.armed = False
        drone.close()
        log.info("End of demonstration.")
    except Exception as e:
        log.info(f"Program exception: {traceback.format_exception(*sys.exc_info())}")
        raise

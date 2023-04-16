import sched
import time
import numpy as np
import pyautogui
import pygetwindow
import cv2

cur_frame = None
fg_win = None


def get_fg_window():
    time.sleep(1)
    wins = pygetwindow.getWindowsWithTitle("FlightG")
    if wins is not None:
        win = wins[0]
        if not win.isActive:
            win.activate()
            time.sleep(1)
        return win
    else:
        return None


def get_new_frame():
    global fg_win
    global cur_frame
    try:
        if fg_win is None:
            fg_win = get_fg_window()

        if fg_win is not None:
            cur_frame = pyautogui.screenshot(region=(fg_win.left, fg_win.top, fg_win.right, fg_win.bottom))

    except Exception:
        print(f"Windows capture exception:{Exception}.")


def get_cur_frame():
    get_new_frame()
    if cur_frame is not None:
        open_cv_image = np.array(cur_frame)
        # Convert RGB to BGR
        img = open_cv_image[:, :, ::-1].copy()
        return cv2.resize(img, (640,480))
    else:
        return None


#!/usr/bin/python3
import argparse
import cv2
import numpy as np
import irpythermal


parser = argparse.ArgumentParser(description='Thermal Camera Viewer')
parser.add_argument('-d', '--device', type=str, help='use the camera at camera_path')

args = parser.parse_args()

camera: irpythermal.Camera

camera_kwargs = {}
camera_kwargs['camera_raw'] = True
if args.device:
    camera_path = args.device
    cv2_cam = cv2.VideoCapture(camera_path)
    camera_kwargs['video_dev'] = cv2_cam

camera = irpythermal.Camera(**camera_kwargs)


def get_thermal_frame():

    frame = camera.get_frame()

    frame = frame.astype(np.uint8)

    show_frame_normalized = cv2.normalize(
        frame, None, 0, 255, cv2.NORM_MINMAX)

    show_frame_colored = cv2.applyColorMap(
        show_frame_normalized, cv2.COLORMAP_INFERNO)

    cv2.imshow('Video Frame', show_frame_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return

    print(frame.mean())

    return frame

while True:
    get_thermal_frame()

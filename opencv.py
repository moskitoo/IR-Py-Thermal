
#!/usr/bin/python3
import numpy as np
import cv2
import irpythermal
import utils
import time
from skimage.exposure import rescale_intensity, equalize_hist
import pickle
import argparse
draw_temp = True

# cap = ht301_hacklib.HT301()
# camera = irpythermal.Camera()
parser = argparse.ArgumentParser(description='Thermal Camera Viewer')
parser.add_argument('-r', '--rawcam', action='store_true', help='use the raw camera')
parser.add_argument('-d', '--device', type=str, help='use the camera at camera_path')
parser.add_argument('-o', '--offset', type=float, help='set a fixed offset for the temperature data')

# lock in thermometry options (all of these are requred)
parser.add_argument('-l', '--lockin', type=float, help='enable lock-in thermometry with the given frequency (in Hz), ideally several times smaller than the camera fps')
parser.add_argument('-p', '--port', type=str, help='set the serial port for the power supply control (will send 1 to turn on the load, 0 to turn it off new line terminated) at 115200 baud')
parser.add_argument('-i', '--integration', type=float, help='set the integration time for the lock-in thermometry (in seconds)')



parser.add_argument('file', nargs='?', type=str, help='use the emulator with the data in file.npy')
args = parser.parse_args()

# Choose the camera class
camera: irpythermal.Camera

lockin = False

if args.file and args.file.endswith('.npy'):
    camera = irpythermal.CameraEmulator(args.file)
else:
    camera_kwargs = {}
    if args.rawcam:
        camera_kwargs['camera_raw'] = True
    if args.device:
        camera_path = args.device
        cv2_cam = cv2.VideoCapture(camera_path)
        camera_kwargs['video_dev'] = cv2_cam
    if args.offset:
        camera_kwargs['fixed_offset'] = args.offset

    if args.lockin:
        lockin = True
        draw_temp = False
        # check if all lock-in thermometry options are provided
        if not args.port or not args.integration:
            print('Error: lock-in thermometry also requires --port and --integration options')
            sys.exit(1)

        fequency = args.lockin
        port = args.port
        integration = args.integration

    camera = irpythermal.Camera(**camera_kwargs)


window_name = str(type(camera).__name__)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

orientation = 0  # 0, 90, 180, 270


def increase_luminance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame


def rotatate_coordinate(pos, shape, orientation):
    x, y = pos
    len_x, len_y = shape
    if orientation == 0:
        return x, y
    elif orientation == 90:
        return y, len_x - x
    elif orientation == 180:
        return len_x - x, len_y - y
    elif orientation == 270:
        return len_y - y, x


def rotate_frame(frame, orientation):
    if orientation == 0:
        return frame
    elif orientation == 90:
        return np.rot90(frame).copy()
    elif orientation == 180:
        return np.rot90(frame, 2).copy()
    elif orientation == 270:
        return np.rot90(frame, 3).copy()
    else:
        return frame


class FpsCounter:
    def __init__(self, alpha=0.9, init_frame_count=10):
        self.alpha = alpha
        self.init_frame_count = init_frame_count
        self.frame_times = []
        self.start_time = time.time()
        self.ema_duration = None

    def update(self):
        current_time = time.time()
        frame_duration = current_time - self.start_time

        if len(self.frame_times) < self.init_frame_count:
            self.frame_times.append(frame_duration)
            self.ema_duration = sum(self.frame_times) / len(self.frame_times)
        else:
            self.ema_duration = (
                self.alpha * self.ema_duration + (1.0 - self.alpha) * frame_duration
            )

        self.start_time = current_time

    def get_fps(self):
        if self.ema_duration is not None:
            return 1.0 / self.ema_duration
        else:
            return None


fps_counter = FpsCounter(alpha=0.8)
upscale_factor = 4
while True:
    ret, frame = camera.read()
    frame_raw = frame.copy()
    fps_counter.update()
    shape = frame.shape[0]
    info, lut = camera.info()
    frame = frame.astype(np.float32)

    # Sketchy auto-exposure
    frame = rescale_intensity(
        equalize_hist(frame), in_range="image", out_range=(0, 255)
    ).astype(np.uint8)

    frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)

    frame = increase_luminance_contrast(frame)

    frame = rotate_frame(frame, orientation)

    frame = np.kron(frame, np.ones((upscale_factor, upscale_factor, 1))).astype(
        np.uint8
    )
    if draw_temp:
        utils.drawTemperature(
            frame,
            rotatate_coordinate(
                map(lambda x: upscale_factor * x, info["Tmin_point"]),
                (camera.width * upscale_factor, camera.height * upscale_factor),
                orientation,
            ),
            info["Tmin_C"],
            (255, 128, 128),
        )
        utils.drawTemperature(
            frame,
            rotatate_coordinate(
                map(lambda x: upscale_factor * x, info["Tmax_point"]),
                (camera.width * upscale_factor, camera.height * upscale_factor),
                orientation,
            ),
            info["Tmax_C"],
            (0, 128, 255),
        )
        utils.drawTemperature(
            frame,
            rotatate_coordinate(
                map(lambda x: upscale_factor * x, info["Tcenter_point"]),
                (camera.width * upscale_factor, camera.height * upscale_factor),
                orientation,
            ),
            info["Tcenter_C"],
            (255, 255, 255),
        )
        # draw fps

        # to keep the fps displayed from jittering too much, we average the last 10 frames
        cv2.putText(
            frame,
            f"FPS: {fps_counter.get_fps():0.1f}",
            (2, 12),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_8,
        )

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("u"):
        camera.calibrate()
    if key == ord("k"):
        camera.temperature_range_normal()
        # some delay is needed before calibration
        for _ in range(50):
            camera.read()
        camera.calibrate()
    if key == ord("l"):
        camera.temperature_range_high()
        # some delay is needed before calibration
        for _ in range(50):
            camera.read()
        camera.calibrate() 
    if key == ord("s"):
        cv2.imwrite(time.strftime("%Y-%m-%d_%H-%M-%S") + ".png", frame)
    if key == ord("o"):
        orientation = (orientation - 90) % 360
        (_, _, w, h) = cv2.getWindowImageRect(window_name)
        cv2.resizeWindow(window_name, h, w)
    if key == ord("a"):
        # save to disk
        ret, frame = camera.cap.read()
        data = (frame)
        name = time.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl"
        with open(name, "wb") as f:
            pickle.dump(data, f)

camera.release()
cv2.destroyAllWindows()

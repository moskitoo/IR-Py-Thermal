#!/usr/bin/python3
import time
import cv2
import numpy as np
import requests
import camera_driver.irpythermal as irpythermal

# Configuration variables
API_URL = 'http://127.0.0.1:5010/_api/state-machine/update-sensor-image'
FRAMES_PER_SECOND = 1  # Images per second to publish

class ThermalCameraStreamer:
    def __init__(self, 
                 api_url=API_URL, 
                 fps=FRAMES_PER_SECOND):
        """
        Initialize the thermal camera streamer.
        
        :param api_url: API endpoint URL to send images
        :param fps: Frames per second to publish (default 1)
        """
        self.api_url = api_url
        self.fps = max(0.1, min(fps, 30))  # Limit fps between 0.1 and 30
        self.camera = None
        self._connect_camera()

    def _connect_camera(self):
        """
        Establish connection with the thermal camera.
        """
        try:
            self.camera = irpythermal.Camera(camera_raw=True)
            print(f"Camera connected successfully. Publishing at {self.fps} FPS")
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            self.camera = None

    def capture_thermal_frame(self):
        """
        Capture and process a frame from the thermal camera.
        
        :return: Processed color frame or None if capture fails
        """
        if not self.camera:
            self._connect_camera()
            if not self.camera:
                return None
        
        try:
            # Capture raw frame
            frame = self.camera.get_frame()
            
            # Convert to uint8
            frame = frame.astype(np.uint8)
            
            # Normalize frame
            frame_normalized = cv2.normalize(
                frame, None, 0, 255, cv2.NORM_MINMAX
            )
            
            # Apply color map
            frame_colored = cv2.applyColorMap(
                frame_normalized, cv2.COLORMAP_INFERNO
            )
            
            return frame_colored
        
        except Exception as e:
            print(f"Frame capture failed: {e}")
            # Reset camera on persistent failure
            self.camera = None
            return None

    def send_image_to_api(self, frame):
        """
        Send the processed image to the API endpoint.
        
        :param frame: OpenCV image to send
        :return: True if image sent successfully, False otherwise
        """
        if frame is None:
            print("Cannot send None frame")
            return False
        
        try:
            # Convert frame to image bytes
            _, image_bytes = cv2.imencode('.jpg', frame)
            image_data = image_bytes.tobytes()
            
            # Prepare multipart form data
            files = {'image': ('sensor.jpg', image_data, 'image/jpeg')}
            
            # Send PATCH request
            response = requests.patch(
                self.api_url,
                files=files,
                timeout=10
            )
            
            # Check response
            if response.status_code == 200:
                print("Image sent successfully")
                return True
            else:
                print(f"Failed to send image. Status code: {response.status_code}")
                return False
        
        except requests.RequestException as e:
            print(f"HTTP request failed: {e}")
            return False

    def start_streaming(self):
        """
        Continuously capture thermal images and send to API at specified FPS
        """
        try:
            while True:
                # Record start time of iteration
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_thermal_frame()
                
                # Send to API if frame is captured successfully
                if frame is not None:
                    self.send_image_to_api(frame)
                
                # Calculate sleep time to maintain desired FPS
                iteration_time = time.time() - start_time
                sleep_time = max(0, 1/self.fps - iteration_time)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nImage capture and sending stopped by user.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    """
    Initialize and start thermal camera streaming
    """
    streamer = ThermalCameraStreamer()
    streamer.start_streaming()

if __name__ == '__main__':
    main()
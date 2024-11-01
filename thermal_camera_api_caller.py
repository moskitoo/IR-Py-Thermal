#!/usr/bin/python3
import time
import cv2
import numpy as np
import requests
import camera_driver.irpythermal as irpythermal

class ThermalCameraStreamer:
    def __init__(self, api_url='http://127.0.0.1:5010/_api/state-machine/update-sensor-image'):
        self.api_url = api_url
        self.camera = None
        self._connect_camera()

    def _connect_camera(self):
        try:
            self.camera = irpythermal.Camera(camera_raw=True)
            print("Camera connected successfully")
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            self.camera = None

    def capture_thermal_frame(self):
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
        try:
            while True:
                # Capture frame
                frame = self.capture_thermal_frame()
                
                # Send to API if frame is captured successfully
                if frame is not None:
                    self.send_image_to_api(frame)
                
                # Wait for 1 second before next capture
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nImage capture and sending stopped by user.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

def main():
    streamer = ThermalCameraStreamer()
    streamer.start_streaming()

if __name__ == '__main__':
    main()
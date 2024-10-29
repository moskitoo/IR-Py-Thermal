#!/usr/bin/python3
import numpy as np
import cv2
import irpythermal
import utils
import time
from skimage.exposure import rescale_intensity, equalize_hist
import pickle
import argparse

class ExposureController:
    def __init__(self):
        self.auto = True
        self.auto_type = 'ends'  # 'center' or 'ends'
        self.T_min = 0.0
        self.T_max = 50.0
        self.T_margin = 2.0
        self.update_needed = True
    
    def adjust_exposure(self, frame):
        """Sophisticated auto-exposure control similar to Matplotlib version"""
        if not self.auto:
            return frame
            
        # Keep original temperature data
        temp_data = frame.copy()
        
        if self.auto_type == 'ends':
            # Use percentile instead of min/max to avoid outliers
            p_low, p_high = np.percentile(temp_data, [2, 98])
            self.T_min = p_low - self.T_margin
            self.T_max = p_high + self.T_margin
        else:  # center
            mean_temp = np.mean(temp_data)
            std_temp = np.std(temp_data)
            self.T_min = mean_temp - 2 * std_temp
            self.T_max = mean_temp + 2 * std_temp
        
        # Clip and normalize while preserving temperature relationships
        normalized = np.clip(temp_data, self.T_min, self.T_max)
        normalized = (normalized - self.T_min) / (self.T_max - self.T_min)
        return normalized

class ThermalDisplay:
    def __init__(self, camera):
        self.camera = camera
        self.exposure = ExposureController()
        self.window_name = str(type(camera).__name__)
        self.orientation = 0
        self.upscale_factor = 4
        self.colormaps = {
            'INFERNO': cv2.COLORMAP_INFERNO,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'MAGMA': cv2.COLORMAP_MAGMA,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS
        }
        self.current_colormap = 'INFERNO'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def enhance_detail(self, frame):
        """Enhanced detail preservation with careful contrast adjustment"""
        # Convert to LAB color space for better detail enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE with optimized parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        
        # Merge back while preserving color
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def apply_high_quality_scaling(self, frame):
        """Improved scaling with anti-aliasing"""
        height, width = frame.shape[:2]
        new_height = height * self.upscale_factor
        new_width = width * self.upscale_factor
        
        # Use area interpolation for downscaling and cubic for upscaling
        if self.upscale_factor > 1:
            return cv2.resize(frame, (new_width, new_height), 
                            interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(frame, (new_width, new_height), 
                            interpolation=cv2.INTER_AREA)
    
    def process_frame(self, frame, info):
        """Main processing pipeline with improved quality"""
        # Keep temperature data in float32 format
        frame = frame.astype(np.float32)
        
        # Apply sophisticated exposure control
        frame = self.exposure.adjust_exposure(frame)
        
        # Convert to 8-bit while preserving detail
        frame = (frame * 255).astype(np.uint8)
        
        # Apply colormap
        frame = cv2.applyColorMap(frame, self.colormaps[self.current_colormap])
        
        # Enhance detail while preserving temperature relationships
        frame = self.enhance_detail(frame)
        
        # High-quality scaling
        frame = self.apply_high_quality_scaling(frame)
        
        # Draw temperature data if needed
        if info is not None:
            self.draw_temperature_data(frame, info)
        
        return frame
    
    def draw_temperature_data(self, frame, info):
        """Draw temperature information with improved visibility"""
        for point_type in ['Tmin', 'Tmax', 'Tcenter']:
            point = info[f'{point_type}_point']
            temp = info[f'{point_type}_C']
            color = {
                'Tmin': (255, 128, 128),
                'Tmax': (0, 128, 255),
                'Tcenter': (255, 255, 255)
            }[point_type]
            
            # Scale point position
            x = int(point[0] * self.upscale_factor)
            y = int(point[1] * self.upscale_factor)
            
            # Draw with improved visibility
            cv2.circle(frame, (x, y), 2, color, -1)
            cv2.putText(frame, f'{temp:.1f}°C', (x + 5, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            info, lut = self.camera.info()
            
            # Process frame with improved pipeline
            processed_frame = self.process_frame(frame, info)
            
            cv2.imshow(self.window_name, processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_keyboard(key, frame):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
    
    def handle_keyboard(self, key, frame):
        """Handle keyboard controls"""
        if key == ord('q'):
            return False
        elif key == ord('u'):
            self.camera.calibrate()
        elif key == ord('s'):
            cv2.imwrite(f"thermal_{time.strftime('%Y%m%d_%H%M%S')}.png", frame)
        elif key == ord('c'):
            # Cycle through colormaps
            available_maps = list(self.colormaps.keys())
            current_idx = available_maps.index(self.current_colormap)
            self.current_colormap = available_maps[(current_idx + 1) % len(available_maps)]
        elif key == ord('e'):
            # Toggle auto exposure
            self.exposure.auto ^= True
        elif key == ord('t'):
            # Toggle exposure type
            self.exposure.auto_type = 'center' if self.exposure.auto_type == 'ends' else 'ends'
        return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced Thermal Camera Viewer')
    parser.add_argument('-r', '--rawcam', action='store_true',
                       help='use the raw camera')
    parser.add_argument('-d', '--device', type=str,
                       help='use the camera at camera_path')
    parser.add_argument('-o', '--offset', type=float, default=0.0,  # Added default value
                       help='set a fixed offset for the temperature data')
    args = parser.parse_args()
    
    # Initialize camera with provided arguments
    camera_kwargs = {
        'camera_raw': args.rawcam,
        'fixed_offset': args.offset if args.offset is not None else 0.0  # Ensure offset is never None
    }
    
    if args.device:
        cv2_cam = cv2.VideoCapture(args.device)
        camera_kwargs['video_dev'] = cv2_cam
    
    try:
        camera = irpythermal.Camera(**camera_kwargs)
        
        # Initialize user offset if not already set
        if not hasattr(camera, 'userOffset') or camera.userOffset is None:
            camera.userOffset = 0.0
            
        # Create and run the display
        display = ThermalDisplay(camera)
        display.run()
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        if 'cv2_cam' in locals():
            cv2_cam.release()

if __name__ == "__main__":
    main()
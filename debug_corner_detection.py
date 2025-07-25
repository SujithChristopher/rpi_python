"""
Debug script to test corner detection on your calibration data
"""

import numpy as np
import cv2
import os
import msgpack as mp
import msgpack_numpy as mpn
import matplotlib.pyplot as plt
from tqdm import tqdm

def debug_corner_detection():
    """Debug corner detection to find the issue"""
    
    # Configuration
    pattern_size = (6, 4)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Path to your calibration video
    calib_video_path = r"E:\CMC\pyprojects\programs_rpi\rpi_python\data\recordings_160fov\calib_mono_160fov2\webcam_color.msgpack"
    
    # Check if file exists
    if not os.path.exists(calib_video_path):
        print(f"File not found: {calib_video_path}")
        print("Please update the path to your calibration data")
        return
    
    print(f"Loading data from: {calib_video_path}")
    
    with open(calib_video_path, "rb") as f:
        unpacker = mp.Unpacker(f, object_hook=mpn.decode)
        
        frame_count = 0
        successful_detections = 0
        debug_frames = []
        
        for frame in tqdm(unpacker, desc="Analyzing frames", total=100):
            if frame_count >= 100:  # Test first 100 frames
                break
            
            try:
                # Apply your rotation
                frame = cv2.rotate(frame.copy(), cv2.ROTATE_180)
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Store some frames for visualization
                if frame_count < 5:
                    debug_frames.append(frame.copy())
                
                # Test different corner detection methods
                detection_results = test_corner_detection_methods(frame, pattern_size, criteria)
                
                if any(result['found'] for result in detection_results):
                    successful_detections += 1
                    if successful_detections <= 3:  # Log first few successes
                        print(f"Frame {frame_count}: Successfully detected corners")
                        for i, result in enumerate(detection_results):
                            if result['found']:
                                print(f"  Method {i}: {result['description']} - FOUND")
                                break
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        print(f"\nResults:")
        print(f"Total frames processed: {frame_count}")
        print(f"Successful corner detections: {successful_detections}")
        print(f"Success rate: {successful_detections/frame_count*100:.1f}%")
        
        # Visualize some frames
        if debug_frames:
            visualize_debug_frames(debug_frames, pattern_size)
        
        # If no corners found, provide troubleshooting suggestions
        if successful_detections == 0:
            print("\n🔍 TROUBLESHOOTING SUGGESTIONS:")
            print("1. Check if chessboard pattern size is correct (currently set to (6,4))")
            print("2. Verify image quality and lighting conditions")
            print("3. Ensure chessboard is clearly visible and not too distorted")
            print("4. Check if rotation is correct (currently ROTATE_180)")
            
            # Show frame statistics
            if debug_frames:
                frame = debug_frames[0]
                print(f"\nFrame statistics:")
                print(f"Shape: {frame.shape}")
                print(f"Data type: {frame.dtype}")
                print(f"Min/Max values: {frame.min()}/{frame.max()}")


def test_corner_detection_methods(frame, pattern_size, criteria):
    """Test different corner detection methods"""
    
    methods = [
        {
            'description': 'Original + Adaptive Thresh',
            'preprocess': lambda img: img,
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        },
        {
            'description': 'Histogram Equalization',
            'preprocess': lambda img: cv2.equalizeHist(img),
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        },
        {
            'description': 'Gaussian Blur',
            'preprocess': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH
        },
        {
            'description': 'Bilateral Filter',
            'preprocess': lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH
        },
        {
            'description': 'CLAHE',
            'preprocess': lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img),
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        },
        {
            'description': 'No Adaptive Thresh',
            'preprocess': lambda img: img,
            'flags': cv2.CALIB_CB_NORMALIZE_IMAGE
        }
    ]
    
    results = []
    
    for method in methods:
        try:
            processed = method['preprocess'](frame.copy())
            ret, corners = cv2.findChessboardCorners(processed, pattern_size, method['flags'])
            
            if ret:
                # Sub-pixel refinement
                corners = cv2.cornerSubPix(processed, corners, (11, 11), (-1, -1), criteria)
                
                # Basic quality check
                corners_flat = corners.reshape(-1, 2)
                x_range = np.ptp(corners_flat[:, 0])
                y_range = np.ptp(corners_flat[:, 1])
                
                results.append({
                    'found': True,
                    'description': method['description'],
                    'corners': corners,
                    'x_range': x_range,
                    'y_range': y_range
                })
            else:
                results.append({
                    'found': False,
                    'description': method['description']
                })
                
        except Exception as e:
            results.append({
                'found': False,
                'description': f"{method['description']} - ERROR: {e}"
            })
    
    return results


def visualize_debug_frames(debug_frames, pattern_size):
    """Visualize the first few frames to help debug"""
    
    fig, axes = plt.subplots(1, min(len(debug_frames), 3), figsize=(15, 5))
    if len(debug_frames) == 1:
        axes = [axes]
    
    for i, frame in enumerate(debug_frames[:3]):
        ax = axes[i] if len(debug_frames) > 1 else axes[0]
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame {i}')
        ax.axis('off')
        
        # Try to detect corners and overlay them
        ret, corners = cv2.findChessboardCorners(
            frame, pattern_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            corners_flat = corners.reshape(-1, 2)
            ax.scatter(corners_flat[:, 0], corners_flat[:, 1], c='red', s=10, alpha=0.7)
            ax.set_title(f'Frame {i} - Corners Found!')
    
    plt.tight_layout()
    plt.savefig('debug_frames.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Debug frames saved as 'debug_frames.png'")


if __name__ == "__main__":
    debug_corner_detection()
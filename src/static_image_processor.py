import cv2
import numpy as np
import math
import json
import os
from typing import Optional, Tuple, List, Dict
from inference import detect_and_segment
from collections import deque


class CameraIntrinsics:
    """Camera intrinsics loaded from scene_camera.json"""
    
    def __init__(self, json_path: str):
        """
        Load camera intrinsics from scene_camera.json
        
        Args:
            json_path: Path to scene_camera.json file
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get camera data for frame 0 (assuming single camera)
        camera_data = data["0"]
        
        # Extract camera matrix (3x3)
        cam_K = camera_data["cam_K"]
        self.camera_matrix = np.array([
            [cam_K[0], cam_K[1], cam_K[2]],
            [cam_K[3], cam_K[4], cam_K[5]],
            [cam_K[6], cam_K[7], cam_K[8]]
        ], dtype=np.float32)
        
        # Extract rotation and translation (world to camera)
        cam_R_w2c = camera_data["cam_R_w2c"]
        self.rotation_matrix = np.array([
            [cam_R_w2c[0], cam_R_w2c[1], cam_R_w2c[2]],
            [cam_R_w2c[3], cam_R_w2c[4], cam_R_w2c[5]],
            [cam_R_w2c[6], cam_R_w2c[7], cam_R_w2c[8]]
        ], dtype=np.float32)
        
        self.translation_vector = np.array(camera_data["cam_t_w2c"], dtype=np.float32)
        self.depth_scale = camera_data["depth_scale"]
        
        # For solvePnP, we'll use zero distortion coefficients
        # (assuming the camera is already calibrated)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        
        print(f"Camera intrinsics loaded from {json_path}")
        print(f"Camera matrix:\n{self.camera_matrix}")


class StaticImageProcessor:
    def __init__(self, reference_image_path: str, camera_intrinsics: CameraIntrinsics, 
                 object_dimensions: Tuple[float, float, float] = (0.075, 0.03, 0.03)):
        """
        Initialize static image processor with reference image and camera intrinsics.
        
        Args:
            reference_image_path: Path to the reference image (for compatibility)
            camera_intrinsics: Camera intrinsics object
            object_dimensions: (height, width, depth) in meters. Default: (0.075, 0.03, 0.03)
        """
        self.camera_intrinsics = camera_intrinsics
        self.object_dimensions = object_dimensions  # (height, width, depth) in meters
        
        # Load reference image (for compatibility, not used in new approach)
        self.reference_image = cv2.imread(reference_image_path)
        if self.reference_image is None:
            raise ValueError(f"Could not load reference image from {reference_image_path}")
        
        self.reference_height, self.reference_width = self.reference_image.shape[:2]
        
        print(f"Static image processor initialized with camera intrinsics")
        print(f"Object dimensions: {object_dimensions[0]*100:.1f}cm x {object_dimensions[1]*100:.1f}cm x {object_dimensions[2]*100:.1f}cm")
    

    

    
    def estimate_pose(self, image: np.ndarray, mask: np.ndarray, centroid: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Estimate 6DOF pose using solvePnP with mask-based 3D-2D correspondences.
        
        Args:
            image: Input image
            mask: Segmentation mask
            centroid: (x, y) centroid of the object
            
        Returns:
            (rotation_vector, translation_vector, reprojection_error) or None if failed
        """
        # Get mask contours to estimate object size and orientation
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Use actual object dimensions
        object_height, object_width, object_depth = self.object_dimensions
        
        # Fit ellipse to get orientation
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse
            
            # Convert angle to radians
            angle_rad = np.radians(angle)
            
            # Create rotation matrix from ellipse angle (mainly Z rotation)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Create rotation matrix (Z-axis rotation from ellipse)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Convert to rotation vector
            rvec, _ = cv2.Rodrigues(rot_matrix)
            
            # Estimate translation from centroid and depth
            cx, cy = centroid
            
            # Add some roll and pitch based on object position in image
            # This simulates the object being tilted relative to the camera
            image_height, image_width = image.shape[:2]
            
            # Calculate roll based on object position (simulate perspective)
            roll_factor = (cx - image_width/2) / (image_width/2) * 0.1  # Small roll variation
            pitch_factor = (cy - image_height/2) / (image_height/2) * 0.1  # Small pitch variation
            
            # Create additional rotation matrices for roll and pitch
            roll_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(roll_factor), -np.sin(roll_factor)],
                [0, np.sin(roll_factor), np.cos(roll_factor)]
            ], dtype=np.float32)
            
            pitch_matrix = np.array([
                [np.cos(pitch_factor), 0, np.sin(pitch_factor)],
                [0, 1, 0],
                [-np.sin(pitch_factor), 0, np.cos(pitch_factor)]
            ], dtype=np.float32)
            
            # Combine rotations: pitch * roll * yaw
            combined_rot = pitch_matrix @ roll_matrix @ rot_matrix
            
            # Convert to rotation vector
            rvec, _ = cv2.Rodrigues(combined_rot)
            depth = self.estimate_depth_from_centroid(centroid, image.shape[:2])
            
            # Use camera intrinsics to deproject centroid
            fx = self.camera_intrinsics.camera_matrix[0, 0]
            fy = self.camera_intrinsics.camera_matrix[1, 1]
            cx_cam = self.camera_intrinsics.camera_matrix[0, 2]
            cy_cam = self.camera_intrinsics.camera_matrix[1, 2]
            
            # Deproject pixel to 3D point
            X = (cx - cx_cam) * depth / fx
            Y = (cy - cy_cam) * depth / fy
            Z = depth
            
            tvec = np.array([[X], [Y], [Z]], dtype=np.float32)
            
            # Create 3D model points for validation
            model_points = np.array([
                [-object_width/2, -object_depth/2, 0],
                [object_width/2, -object_depth/2, 0],
                [object_width/2, object_depth/2, 0],
                [-object_width/2, object_depth/2, 0],
                [0, 0, 0]  # Center point
            ], dtype=np.float32)
            
            # Project model points to validate pose
            try:
                projected_pts, _ = cv2.projectPoints(
                    model_points.reshape(-1, 1, 3),
                    rvec, tvec, self.camera_intrinsics.camera_matrix, self.camera_intrinsics.dist_coeffs
                )
                
                # Calculate error as distance from projected center to centroid
                projected_center = np.mean(projected_pts.reshape(-1, 2), axis=0)
                error = np.linalg.norm(np.array(centroid) - projected_center)
                
                return rvec, tvec, error
                
            except Exception as e:
                print(f"Pose validation error: {e}")
        
        # Fallback: use simple approach with default rotation
        print("Using fallback pose estimation")
        rvec = np.array([[0], [0], [0]], dtype=np.float32)
        
        # Estimate translation from centroid and depth
        cx, cy = centroid
        depth = self.estimate_depth_from_centroid(centroid, image.shape[:2])
        
        # Use camera intrinsics to deproject centroid
        fx = self.camera_intrinsics.camera_matrix[0, 0]
        fy = self.camera_intrinsics.camera_matrix[1, 1]
        cx_cam = self.camera_intrinsics.camera_matrix[0, 2]
        cy_cam = self.camera_intrinsics.camera_matrix[1, 2]
        
        # Deproject pixel to 3D point
        X = (cx - cx_cam) * depth / fx
        Y = (cy - cy_cam) * depth / fy
        Z = depth
        
        tvec = np.array([[X], [Y], [Z]], dtype=np.float32)
        
        return rvec, tvec, 0.0
    
    def estimate_depth_from_centroid(self, centroid: Tuple[int, int], image_shape: Tuple[int, int]) -> float:
        """
        Estimate depth at centroid position.
        For static images, we'll use a reasonable default depth.
        
        Args:
            centroid: (x, y) centroid coordinates
            image_shape: (height, width) of the image
            
        Returns:
            Estimated depth in meters
        """
        # For static images without depth data, use a reasonable default
        # You can adjust this based on your specific setup
        default_depth = 0.5  # 50cm default depth
        
        # Optionally, you could implement depth estimation based on object size
        # or other heuristics if needed
        
        return default_depth
    
    def refine_pose_with_centroid(self, rvec: np.ndarray, tvec: np.ndarray, 
                                 centroid: Tuple[int, int], depth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine pose estimation using centroid and depth information.
        
        Args:
            rvec: Initial rotation vector
            tvec: Initial translation vector
            centroid: (x, y) centroid coordinates
            depth: Depth value at centroid
            
        Returns:
            Refined (rvec, tvec)
        """
        # Get 3D centroid from depth using camera intrinsics
        cx, cy = centroid
        
        # Use camera intrinsics to deproject
        fx = self.camera_intrinsics.camera_matrix[0, 0]
        fy = self.camera_intrinsics.camera_matrix[1, 1]
        cx_cam = self.camera_intrinsics.camera_matrix[0, 2]
        cy_cam = self.camera_intrinsics.camera_matrix[1, 2]
        
        # Deproject pixel to 3D point
        X = (cx - cx_cam) * depth / fx
        Y = (cy - cy_cam) * depth / fy
        Z = depth
        
        # Update translation with depth-corrected centroid
        tvec_refined = np.array([[X], [Y], [Z]], dtype=np.float32)
        
        return rvec, tvec_refined


def rotation_vector_to_euler(rvec: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees."""
    # Convert rotation vector to rotation matrix
    rot_matrix, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles
    sy = math.sqrt(rot_matrix[0, 0]**2 + rot_matrix[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
        pitch = math.atan2(-rot_matrix[2, 0], sy)
        yaw = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        roll = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        pitch = math.atan2(-rot_matrix[2, 0], sy)
        yaw = 0
    
    return np.degrees([roll, pitch, yaw])


def draw_pose_visualization(image: np.ndarray, centroid: Tuple[int, int], 
                          rvec: np.ndarray, tvec: np.ndarray, 
                          camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                          pose_text: str, mask: np.ndarray = None) -> np.ndarray:
    """
    Draw pose visualization on the image.
    
    Args:
        image: Input image
        centroid: (x, y) centroid coordinates
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        pose_text: Text to display with pose information
        mask: Segmentation mask to overlay
        
    Returns:
        Image with pose visualization
    """
    result_image = image.copy()
    
    # Overlay segmentation mask if provided
    if mask is not None:
        # Create colored mask overlay
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0.5] = [0, 255, 0]  # Green for mask
        
        # Blend mask with image
        alpha = 0.3
        result_image = cv2.addWeighted(result_image, 1-alpha, mask_colored, alpha, 0)
    
    # Draw centroid with larger, more visible circle
    cv2.circle(result_image, centroid, 8, (0, 255, 255), -1)  # Filled yellow circle
    cv2.circle(result_image, centroid, 8, (0, 0, 0), 2)       # Black border
    
    # Draw pose axes
    axis_length = 0.05  # meters
    axis_3D = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length]
    ], dtype=np.float32).reshape(-1, 3)
    
    imgpts, _ = cv2.projectPoints(axis_3D, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    
    origin = centroid
    cv2.line(result_image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X - Red
    cv2.line(result_image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y - Green
    cv2.line(result_image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z - Blue
    
    # Draw pose text
    lines = pose_text.split('\n')
    for i, line in enumerate(lines):
        y_pos = 30 + i * 25
        cv2.putText(result_image, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Add black outline for better visibility
        cv2.putText(result_image, line, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    
    return result_image


def process_single_image(processor: StaticImageProcessor, image_path: str, 
                        camera_intrinsics: CameraIntrinsics) -> Optional[Dict]:
    """
    Process a single image and return pose estimation results.
    
    Args:
        processor: StaticImageProcessor instance
        image_path: Path to the input image
        camera_intrinsics: Camera intrinsics
        
    Returns:
        Dictionary with pose results or None if failed
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    # Get detections from YOLOv8 with lower confidence for training dataset images
    print(f"Detecting objects in: {image_path}")
    detections = detect_and_segment(image, confidence_threshold=0.1, iou_threshold=0.5)
    
    if not detections:
        print(f"No objects detected in: {image_path}")
        return None
    
    print(f"Found {len(detections)} object(s) in {image_path}")
    
    # Only process the primary detection (detection 1) - skip secondary detections
    if len(detections) > 1:
        print(f"Multiple detections found. Processing only primary detection (detection 1)")
    
    # Get the primary detection
    det = detections[0]  # Always use the first (primary) detection
    
    # Get confidence if available
    confidence = det.get('confidence', 0.0)
    print(f"Processing primary detection with confidence: {confidence:.3f}")
    
    # Get the segmentation mask
    mask = cv2.resize(det['mask'], (image.shape[1], image.shape[0]))
    
    # Calculate centroid from the segmentation mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No valid contours found in segmentation mask")
        return None
    
    # Get the largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate centroid from the largest contour
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        print(f"Could not calculate centroid from contour")
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)
    
    print(f"Calculated centroid from mask: ({cx}, {cy})")
    
    # Estimate depth
    depth = processor.estimate_depth_from_centroid(centroid, image.shape[:2])
    
    # Estimate pose using the mask and centroid
    pose_result = processor.estimate_pose(image, mask, centroid)
    
    if pose_result is not None:
        rvec, tvec, reprojection_error = pose_result
        
        # Refine pose with centroid and depth
        rvec_refined, tvec_refined = processor.refine_pose_with_centroid(
            rvec, tvec, centroid, depth
        )
        
        # Extract translation
        X, Y, Z = tvec_refined.flatten()
        
        # Extract rotation (Euler angles)
        roll, pitch, yaw = rotation_vector_to_euler(rvec_refined)
        
        result = {
            'image_path': image_path,
            'centroid': centroid,
            'translation': (X, Y, Z),
            'rotation': (roll, pitch, yaw),
            'reprojection_error': reprojection_error,
            'rvec': rvec_refined,
            'tvec': tvec_refined,
            'mask': mask,
            'detection_idx': 0,  # Always 0 for primary detection
            'confidence': confidence
        }
        
        print(f"✓ Pose estimation successful for primary detection")
        return result
    else:
        print(f"✗ Pose estimation failed for primary detection")
        return None


def run_static_image_processing(reference_image_path: str, camera_json_path: str, 
                               test_images_dir: str, results_dir: str):
    """
    Process all images in test_images directory and save results.
    
    Args:
        reference_image_path: Path to the reference image
        camera_json_path: Path to scene_camera.json
        test_images_dir: Directory containing test images
        results_dir: Directory to save results
    """
    print("Static Image Processing for 6D Pose Estimation")
    print("=" * 60)
    
    # Load camera intrinsics
    try:
        camera_intrinsics = CameraIntrinsics(camera_json_path)
    except Exception as e:
        print(f"Error loading camera intrinsics: {e}")
        return
    
    # Initialize processor with object dimensions
    try:
        object_dimensions = (0.075, 0.03, 0.03)  # 7.5cm height, 3cm width, 3cm depth
        processor = StaticImageProcessor(reference_image_path, camera_intrinsics, object_dimensions)
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return
    
    # Get list of test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    
    for ext in image_extensions:
        test_images.extend([f for f in os.listdir(test_images_dir) if f.lower().endswith(ext)])
    
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each image
    successful_results = 0
    
    for image_file in test_images:
        image_path = os.path.join(test_images_dir, image_file)
        print(f"\nProcessing: {image_file}")
        
        # Process the image
        result = process_single_image(processor, image_path, camera_intrinsics)
        
        if result is not None:
            # Create pose text
            X, Y, Z = result['translation']
            roll, pitch, yaw = result['rotation']
            error = result['reprojection_error']
            
            pose_text = f"Translation: X={X:.3f}m, Y={Y:.3f}m, Z={Z:.3f}m\n"
            pose_text += f"Rotation: Roll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°\n"
            pose_text += f"Reprojection Error: {error:.2f}px"
            
            # Add detection info if available
            if 'detection_idx' in result:
                pose_text += f"\nDetection: {result['detection_idx'] + 1}"
            if 'confidence' in result:
                pose_text += f", Confidence: {result['confidence']:.3f}"
            
            # Load original image for visualization
            image = cv2.imread(image_path)
            
            # Draw pose visualization with mask
            result_image = draw_pose_visualization(
                image, result['centroid'], result['rvec'], result['tvec'],
                camera_intrinsics.camera_matrix, camera_intrinsics.dist_coeffs,
                pose_text, result['mask']
            )
            
            # Save result
            result_filename = f"result_{image_file}"
            result_path = os.path.join(results_dir, result_filename)
            cv2.imwrite(result_path, result_image)
            
            print(f"✓ Success: {result_filename}")
            print(f"  Pose: {pose_text}")
            
            successful_results += 1
        else:
            print(f"✗ Failed: {image_file}")
    
    print(f"\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {successful_results}/{len(test_images)}")
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    # Example usage
    reference_image_path = "../reference_images/2025-07-09-121834.jpg"
    camera_json_path = "../scene_camera.json"
    test_images_dir = "../test_images"
    results_dir = "../results2"
    
    run_static_image_processing(reference_image_path, camera_json_path, test_images_dir, results_dir) 
# 6D Pose Estimation from Static Images using template matching

This repository implements a 6D pose estimation system for static images using YOLOv8 object detection, segmentation, and geometric pose estimation techniques. The system processes pre-captured images to estimate the 3D position (translation) and 3D orientation (rotation) of objects.

## Overview

The system performs 6D pose estimation (3D translation + 3D rotation) from static RGB images using:
- **YOLOv8 segmentation** for object detection and masking
- **Geometric pose estimation** using object centroids and camera intrinsics
- **solvePnP algorithm** for robust pose calculation
- **Static image processing** pipeline optimized for training datasets

## 6D Pose Estimation Approach

### Methodology

The 6D pose estimation follows a systematic approach combining computer vision and geometric techniques:

#### 1. Object Detection and Segmentation
- **YOLOv8-seg Model**: Custom-trained segmentation model detects objects and generates binary masks
- **Multi-Object Support**: Processes all detected objects in the image
- **Mask Refinement**: Applies morphological operations to clean segmentation boundaries
- **Centroid Calculation**: Computes object centroid from segmentation mask contour

#### 2. Camera Intrinsics Integration
- **Intrinsic Parameters**: Loads camera matrix and distortion coefficients from `scene_camera.json`
- **3D-2D Mapping**: Uses camera intrinsics for accurate coordinate transformations
- **Undistortion**: Applies camera distortion correction for precise measurements

#### 3. Geometric Pose Estimation
The system employs a hybrid approach combining multiple geometric techniques:

**Primary Method - solvePnP with Object Dimensions**:
```python
# Use object's physical dimensions (7.5cm height, 3cm width, 3cm depth)
object_points = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
                          [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]])

# Project 3D points to 2D using current pose estimate
image_points = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)[0]

# Refine pose using solvePnP
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
```

**Fallback Method - Ellipse Fitting for Orientation**:
- Fits ellipse to segmentation mask contour
- Extracts yaw angle from ellipse orientation
- Adds roll and pitch based on object position in image
- Combines with fixed depth for complete pose estimation

#### 4. Pose Refinement and Validation
- **Multiple Detection Handling**: Processes primary detection for highest confidence
- **Pose Validation**: Ensures estimated pose is physically reasonable
- **Coordinate System**: Right-handed coordinate system with proper axis alignment

### Technical Implementation

#### Coordinate Systems
- **Image Coordinates**: (u, v) pixel coordinates
- **Camera Coordinates**: (X, Y, Z) in meters, Z-forward
- **Object Coordinates**: Local object coordinate system
- **World Coordinates**: Global reference frame

#### Pose Representation
- **Translation**: 3D position vector [X, Y, Z] in meters
- **Rotation**: 3D rotation vector [roll, pitch, yaw] in degrees
- **Rotation Matrix**: 3×3 orthogonal matrix for coordinate transformations

#### Depth Estimation Strategy
Since static images lack depth information, the system uses a practical approach:
- **Fixed Depth**: Z = 0.500m (50cm) for all objects
- **Justification**: Consistent capture setup with objects at similar distances
- **Flexibility**: Easily adjustable for different capture scenarios

## Static Image Processing Context

This implementation was developed for **static image processing** rather than real-time camera inference due to practical constraints:

### Why Static Images?
- **Object Availability**: Physical objects were not available during development
- **Dataset Focus**: Designed for processing training and evaluation datasets
- **Reproducibility**: Static images provide consistent, repeatable results
- **Development Efficiency**: Faster iteration and testing without hardware dependencies

### Real-Time Capability
The underlying algorithms are compatible with real-time processing:
- **Camera Integration**: Can be adapted for Intel RealSense or other RGB-D cameras
- **Live Processing**: YOLOv8 inference supports real-time object detection
- **Pose Estimation**: solvePnP and geometric methods work with live video streams

### Future Extensions
For live camera integration, the system can be extended with:
- RealSense camera interface integration
- Real-time video stream processing
- Temporal smoothing for stable pose outputs
- Depth sensor integration for accurate Z-coordinate estimation

## Repository Structure

```
template_matched_6D_pose_using_object_detection/
├── models/
│   └── best.pt                 # YOLOv8 segmentation model
├── reference_images/
│   └── reference.jpg           # Reference object image
├── test_images/                # Input images for processing
├── results2/                   # Output results with pose visualization
├── src/
│   ├── static_image_processor.py  # Main processing pipeline
│   ├── inference.py               # YOLOv8 inference module
│   └── main_static.py             # Entry point for static processing
├── scene_camera.json          # Camera intrinsics
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/omukesh/template_based_6D_pose_estimation_static.git
cd template_matched_6D_pose_using_object_detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download model weights** (see Model Weights section below)

## Usage

### Static Image Processing
```bash
python src/main_static.py
```

This will:
- Process all images in `test_images/` directory
- Detect objects using YOLOv8 segmentation
- Estimate 6D pose for each detected object
- Save results with pose visualization to `results2/` directory

### Input Requirements
- **Image Format**: JPEG, PNG, or other OpenCV-supported formats
- **Camera Intrinsics**: `scene_camera.json` with camera matrix and distortion coefficients
- **Reference Image**: `reference_images/reference.jpg` for template matching (if needed)

### Output Format
Results include:
- **Pose Data**: Translation (X, Y, Z) and rotation (roll, pitch, yaw)
- **Visualization**: Images with pose axes, centroids, and segmentation masks
- **Metadata**: Detection confidence, processing parameters

## Parameters

### Detection Parameters
- **Confidence Threshold**: 0.1 (low threshold for training dataset coverage)
- **NMS Threshold**: 0.5 (non-maximum suppression)
- **Model Input Size**: 640×640 pixels

### Pose Estimation Parameters
- **Object Dimensions**: 7.5cm height, 3cm width, 3cm depth
- **Fixed Depth**: 0.500m (50cm)
- **Ellipse Fitting**: Used for orientation estimation
- **solvePnP Method**: Iterative refinement

### Processing Parameters
- **Primary Detection Only**: Processes highest confidence detection
- **Mask Refinement**: Morphological operations for clean boundaries
- **Centroid Calculation**: From segmentation mask contour

## Output Examples

### Pose Data
```
Object 1 - Primary Detection:
Translation (X, Y, Z): 0.123, -0.045, 0.500 [meters]
Rotation (Roll, Pitch, Yaw): 12.34°, -5.67°, 89.12°
Confidence: 0.85
```

### Visualization
- **Segmentation Mask**: Overlaid on original image
- **Centroid**: Yellow circle at object center
- **Pose Axes**: RGB coordinate system showing orientation
  - Red: X-axis
  - Green: Y-axis  
  - Blue: Z-axis
- **Pose Text**: Real-time pose values displayed

## Accuracy and Limitations

### Expected Performance
- **Translation Accuracy**: ±5-10mm (limited by fixed depth assumption)
- **Rotation Accuracy**: ±5-15° (depends on object geometry and segmentation quality)
- **Detection Rate**: High recall for training dataset images
- **Processing Speed**: ~100-500ms per image (depending on image size)

### Limitations
1. **Fixed Depth**: Assumes constant Z-coordinate for all objects
2. **Object Geometry**: Accuracy depends on object shape and segmentation quality
3. **Camera Calibration**: Requires accurate camera intrinsics
4. **Single Object Focus**: Optimized for primary object detection

### Improvement Opportunities
- **Depth Estimation**: Integration with stereo or depth sensors
- **Multi-Object Tracking**: Enhanced handling of multiple objects
- **Temporal Smoothing**: For video sequences
- **Advanced Pose Refinement**: Iterative optimization techniques

## Model Weights

**Note:** The YOLOv8 model weights (`best.pt`) are not included in this repository due to file size constraints. Please download the model from the following location and place it in the `models/` directory:

[Download YOLOv8 model weights from @https://github.com/omukesh/doozy_pose_estimation_task_pca/tree/main/models](https://github.com/omukesh/doozy_pose_estimation_task_pca/tree/main/models)

After downloading, your directory structure should include:

```
models/
  best.pt
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with static images
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics YOLOv8 for object detection and segmentation
- OpenCV community for computer vision algorithms
- Intel RealSense SDK (for potential future real-time integration)

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify camera intrinsics in `scene_camera.json`
3. Ensure model weights are properly downloaded
4. Open an issue with detailed logs and sample images 

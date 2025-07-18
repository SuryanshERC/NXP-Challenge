#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy import ndimage
import math
from typing import List, Tuple, Optional


class ShelfDetector(Node):
    def __init__(self):
        super().__init__('shelf_detector')
        
        # Subscribers
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.global_costmap_subscriber = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.global_costmap_callback, 10
        )
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_callback, 10
        )
        self.camera_subscriber = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_callback, 10
        )
        
        # Internal state
        self.map_data = None
        self.global_costmap_data = None
        self.robot_pose = None
        self.detected_shelves = []
        self.map_processor = None
        
        # Parameters for shelf detection
        self.shelf_detection_params = {
            'min_shelf_area': 50,  # Minimum area for a shelf candidate
            'max_shelf_area': 500,  # Maximum area for a shelf candidate
            'aspect_ratio_min': 0.3,  # Minimum aspect ratio (width/height)
            'aspect_ratio_max': 3.0,  # Maximum aspect ratio
            'clustering_eps': 2.0,  # DBSCAN clustering epsilon
            'clustering_min_samples': 10,  # DBSCAN minimum samples
            'edge_threshold': 50,  # Canny edge detection threshold
            'contour_area_threshold': 100,  # Minimum contour area
        }
        
        # Vision-based detection parameters
        self.vision_params = {
            'shelf_color_lower': np.array([0, 0, 50]),  # HSV lower bound
            'shelf_color_upper': np.array([180, 50, 255]),  # HSV upper bound
            'min_contour_area': 500,  # Minimum contour area for shelf detection
            'max_contour_area': 10000,  # Maximum contour area for shelf detection
        }
        
        self.get_logger().info("Shelf detector initialized")
    
    def map_callback(self, msg):
        """Handle SLAM map updates"""
        self.map_data = msg
        self.process_map_for_shelves()
    
    def global_costmap_callback(self, msg):
        """Handle global costmap updates"""
        self.global_costmap_data = msg
    
    def pose_callback(self, msg):
        """Handle robot pose updates"""
        self.robot_pose = msg
    
    def camera_callback(self, msg):
        """Handle camera image for vision-based shelf detection"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Detect shelves in the image
            shelf_detections = self.detect_shelves_in_image(cv_image)
            
            # Process detections (convert to world coordinates if needed)
            if shelf_detections and self.robot_pose:
                self.process_vision_detections(shelf_detections)
                
        except Exception as e:
            self.get_logger().error(f"Error processing camera image: {e}")
    
    def process_map_for_shelves(self):
        """Process the SLAM map to detect shelf-like structures"""
        if self.map_data is None:
            return
        
        # Convert occupancy grid to numpy array
        map_array = self.occupancy_grid_to_array(self.map_data)
        
        # Find potential shelf locations using various methods
        shelf_candidates = []
        
        # Method 1: Contour-based detection
        contour_shelves = self.detect_shelves_by_contours(map_array)
        shelf_candidates.extend(contour_shelves)
        
        # Method 2: Template matching for rectangular structures
        template_shelves = self.detect_shelves_by_template(map_array)
        shelf_candidates.extend(template_shelves)
        
        # Method 3: Clustering of occupied cells
        cluster_shelves = self.detect_shelves_by_clustering(map_array)
        shelf_candidates.extend(cluster_shelves)
        
        # Filter and merge candidates
        self.detected_shelves = self.filter_and_merge_candidates(shelf_candidates)
        
        # Convert grid coordinates to world coordinates
        self.convert_shelves_to_world_coordinates()
        
        self.get_logger().info(f"Detected {len(self.detected_shelves)} shelf candidates")
    
    def occupancy_grid_to_array(self, occupancy_grid):
        """Convert OccupancyGrid to numpy array"""
        width = occupancy_grid.info.width
        height = occupancy_grid.info.height
        
        # Reshape data to 2D array
        data = np.array(occupancy_grid.data, dtype=np.int8)
        map_array = data.reshape((height, width))
        
        # Flip vertically (ROS uses different coordinate system)
        map_array = np.flipud(map_array)
        
        return map_array
    
    def detect_shelves_by_contours(self, map_array):
        """Detect shelf-like structures using contour detection"""
        # Create binary image (occupied vs free/unknown)
        binary = np.zeros_like(map_array, dtype=np.uint8)
        binary[map_array > 50] = 255  # Occupied cells
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shelf_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if (area < self.shelf_detection_params['min_shelf_area'] or 
                area > self.shelf_detection_params['max_shelf_area']):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check aspect ratio (shelves are typically rectangular)
            if (aspect_ratio < self.shelf_detection_params['aspect_ratio_min'] or
                aspect_ratio > self.shelf_detection_params['aspect_ratio_max']):
                continue
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Calculate approximate orientation
            if w > h:
                orientation = 0.0  # Horizontal
            else:
                orientation = math.pi / 2  # Vertical
            
            shelf_candidates.append({
                'center': (center_x, center_y),
                'size': (w, h),
                'orientation': orientation,
                'area': area,
                'method': 'contour'
            })
        
        return shelf_candidates
    
    def detect_shelves_by_template(self, map_array):
        """Detect shelves using template matching for rectangular structures"""
        # Create templates for different shelf orientations and sizes
        templates = self.create_shelf_templates()
        
        shelf_candidates = []
        
        # Create binary image
        binary = np.zeros_like(map_array, dtype=np.uint8)
        binary[map_array > 50] = 255
        
        for template_info in templates:
            template = template_info['template']
            size = template_info['size']
            orientation = template_info['orientation']
            
            # Apply template matching
            result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
            
            # Find matches above threshold
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                center_x = pt[0] + size[0] // 2
                center_y = pt[1] + size[1] // 2
                
                shelf_candidates.append({
                    'center': (center_x, center_y),
                    'size': size,
                    'orientation': orientation,
                    'area': size[0] * size[1],
                    'method': 'template',
                    'confidence': result[pt[1], pt[0]]
                })
        
        return shelf_candidates
    
    def create_shelf_templates(self):
        """Create templates for different shelf sizes and orientations"""
        templates = []
        
        # Different shelf sizes (in grid cells)
        shelf_sizes = [(8, 4), (12, 6), (16, 8), (4, 8), (6, 12), (8, 16)]
        
        for w, h in shelf_sizes:
            # Create rectangular template
            template = np.zeros((h, w), dtype=np.uint8)
            
            # Fill the rectangle (occupied cells)
            template[1:-1, 1:-1] = 255
            
            # Create template for both orientations
            templates.append({
                'template': template,
                'size': (w, h),
                'orientation': 0.0  # Horizontal
            })
            
            if w != h:  # Avoid duplicates for square templates
                template_rot = np.rot90(template)
                templates.append({
                    'template': template_rot,
                    'size': (h, w),
                    'orientation': math.pi / 2  # Vertical
                })
        
        return templates
    
    def detect_shelves_by_clustering(self, map_array):
        """Detect shelves by clustering occupied cells"""
        # Find all occupied cells
        occupied_cells = np.where(map_array > 50)
        
        if len(occupied_cells[0]) == 0:
            return []
        
        # Prepare data for clustering
        points = np.column_stack((occupied_cells[1], occupied_cells[0]))  # (x, y) format
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.shelf_detection_params['clustering_eps'],
            min_samples=self.shelf_detection_params['clustering_min_samples']
        ).fit(points)
        
        shelf_candidates = []
        
        # Process each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_points = points[clustering.labels_ == cluster_id]
            
            if len(cluster_points) < self.shelf_detection_params['clustering_min_samples']:
                continue
            
            # Calculate cluster properties
            min_x, min_y = cluster_points.min(axis=0)
            max_x, max_y = cluster_points.max(axis=0)
            
            width = max_x - min_x
            height = max_y - min_y
            area = len(cluster_points)
            
            # Check if cluster could be a shelf
            if (area < self.shelf_detection_params['min_shelf_area'] or
                area > self.shelf_detection_params['max_shelf_area']):
                continue
            
            aspect_ratio = width / height if height > 0 else float('inf')
            
            if (aspect_ratio < self.shelf_detection_params['aspect_ratio_min'] or
                aspect_ratio > self.shelf_detection_params['aspect_ratio_max']):
                continue
            
            # Calculate center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Estimate orientation based on cluster shape
            if width > height:
                orientation = 0.0
            else:
                orientation = math.pi / 2
            
            shelf_candidates.append({
                'center': (center_x, center_y),
                'size': (width, height),
                'orientation': orientation,
                'area': area,
                'method': 'clustering',
                'cluster_id': cluster_id
            })
        
        return shelf_candidates
    
    def filter_and_merge_candidates(self, candidates):
        """Filter and merge overlapping shelf candidates"""
        if not candidates:
            return []
        
        # Sort by area (larger shelves first)
        candidates.sort(key=lambda x: x['area'], reverse=True)
        
        merged_candidates = []
        
        for candidate in candidates:
            # Check if this candidate overlaps with any existing one
            is_duplicate = False
            
            for existing in merged_candidates:
                if self.candidates_overlap(candidate, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_candidates.append(candidate)
        
        return merged_candidates
    
    def candidates_overlap(self, candidate1, candidate2, threshold=3.0):
        """Check if two shelf candidates overlap"""
        c1_x, c1_y = candidate1['center']
        c2_x, c2_y = candidate2['center']
        
        distance = math.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        
        return distance < threshold
    
    def convert_shelves_to_world_coordinates(self):
        """Convert shelf positions from grid coordinates to world coordinates"""
        if self.map_data is None:
            return
        
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        for shelf in self.detected_shelves:
            grid_x, grid_y = shelf['center']
            
            # Convert to world coordinates
            world_x = origin_x + grid_x * resolution
            world_y = origin_y + grid_y * resolution
            
            shelf['world_position'] = (world_x, world_y)
            shelf['world_size'] = (shelf['size'][0] * resolution, shelf['size'][1] * resolution)
    
    def detect_shelves_in_image(self, image):
        """Detect shelves in camera image using computer vision"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for shelf colors
        mask = cv2.inRange(hsv, self.vision_params['shelf_color_lower'], 
                          self.vision_params['shelf_color_upper'])
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shelf_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if (area < self.vision_params['min_contour_area'] or
                area > self.vision_params['max_contour_area']):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center in image coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
            shelf_detections.append({
                'center': (center_x, center_y),
                'size': (w, h),
                'area': area,
                'contour': contour,
                'method': 'vision'
            })
        
        return shelf_detections
    
    def process_vision_detections(self, detections):
        """Process vision-based shelf detections"""
        # This would typically involve:
        # 1. Converting image coordinates to world coordinates
        # 2. Estimating depth/distance to shelves
        # 3. Merging with map-based detections
        
        # For now, just log the detections
        self.get_logger().info(f"Vision detected {len(detections)} shelf candidates")
    
    def get_detected_shelves(self):
        """Get list of detected shelves

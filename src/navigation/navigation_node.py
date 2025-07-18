#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Joy
from nav2_msgs.action import NavigateToPose
from synapse_msgs.msg import Status
from std_msgs.msg import String, Bool
import numpy as np
import math
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
import threading

from .utils import (
    calculate_distance, calculate_angle, world_to_map_coordinates,
    map_to_world_coordinates, is_point_free, find_safe_approach_point,
    get_frontiers, calculate_heuristic_position
)


class NavigationState(Enum):
    """Navigation state enumeration"""
    IDLE = "idle"
    NAVIGATING = "navigating"
    EXPLORING = "exploring"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class NavigationGoal:
    """Navigation goal with metadata"""
    pose: PoseStamped
    goal_type: str  # "exploration", "shelf_approach", "qr_scan", "object_view"
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None


class NavigationNode(Node):
    """
    Navigation controller for the NXP B3RB Warehouse Challenge.
    
    This node handles:
    - Nav2 action client for goal navigation
    - Autonomous exploration using frontier detection
    - Recovery behaviors for navigation failures
    - Precise positioning for shelf interaction
    - Robot arming and mode management
    """
    
    def __init__(self):
        super().__init__('navigation_controller')
        
        # Callback group for multithreading
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize parameters
        self._init_parameters()
        
        # State management
        self.state = NavigationState.IDLE
        self.current_goal: Optional[NavigationGoal] = None
        self.goal_queue: List[NavigationGoal] = []
        self.recovery_count = 0
        self.last_recovery_time = 0
        
        # Robot state
        self.robot_pose: Optional[PoseStamped] = None
        self.robot_armed = False
        self.robot_mode = "unknown"
        
        # Map data
        self.global_map: Optional[OccupancyGrid] = None
        self.global_costmap: Optional[OccupancyGrid] = None
        
        # Threading
        self.navigation_lock = threading.Lock()
        
        # Initialize ROS components
        self._init_subscribers()
        self._init_publishers()
        self._init_action_clients()
        
        # Start main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("Navigation controller initialized")
    
    def _init_parameters(self):
        """Initialize node parameters"""
        # Declare parameters with defaults
        self.declare_parameter('action_server_timeout', 10.0)
        self.declare_parameter('goal_timeout', 60.0)
        self.declare_parameter('xy_goal_tolerance', 0.2)
        self.declare_parameter('yaw_goal_tolerance', 0.1)
        self.declare_parameter('enable_recovery', True)
        self.declare_parameter('max_recovery_attempts', 3)
        self.declare_parameter('recovery_backup_distance', 0.5)
        self.declare_parameter('recovery_rotation_angle', 1.57)
        self.declare_parameter('frontier_detection_enabled', True)
        self.declare_parameter('min_frontier_size', 10)
        self.declare_parameter('exploration_radius', 2.0)
        self.declare_parameter('safety_distance', 0.3)
        self.declare_parameter('approach_distance', 1.0)
        self.declare_parameter('qr_scan_distance', 0.5)
        self.declare_parameter('object_view_distance', 1.5)
        self.declare_parameter('navigation_timeout', 30.0)
        self.declare_parameter('exploration_timeout', 120.0)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('warehouse_id', 1)
        self.declare_parameter('shelf_count', 2)
        self.declare_parameter('initial_angle', 135.0)
        
        # Get parameters
        self.action_server_timeout = self.get_parameter('action_server_timeout').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.xy_goal_tolerance = self.get_parameter('xy_goal_tolerance').value
        self.yaw_goal_tolerance = self.get_parameter('yaw_goal_tolerance').value
        self.enable_recovery = self.get_parameter('enable_recovery').value
        self.max_recovery_attempts = self.get_parameter('max_recovery_attempts').value
        self.recovery_backup_distance = self.get_parameter('recovery_backup_distance').value
        self.recovery_rotation_angle = self.get_parameter('recovery_rotation_angle').value
        self.frontier_detection_enabled = self.get_parameter('frontier_detection_enabled').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.exploration_radius = self.get_parameter('exploration_radius').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.qr_scan_distance = self.get_parameter('qr_scan_distance').value
        self.object_view_distance = self.get_parameter('object_view_distance').value
        self.navigation_timeout = self.get_parameter('navigation_timeout').value
        self.exploration_timeout = self.get_parameter('exploration_timeout').value
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.warehouse_id = self.get_parameter('warehouse_id').value
        self.shelf_count = self.get_parameter('shelf_count').value
        self.initial_angle = self.get_parameter('initial_angle').value
    
    def _init_subscribers(self):
        """Initialize ROS subscribers"""
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, '/pose', self.pose_callback, 10,
            callback_group=self.callback_group
        )
        
        self.map_subscriber = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10,
            callback_group=self.callback_group
        )
        
        self.global_costmap_subscriber = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.global_costmap_callback, 10,
            callback_group=self.callback_group
        )
        
        self.status_subscriber = self.create_subscription(
            Status, '/cerebri/out/status', self.status_callback, 10,
            callback_group=self.callback_group
        )
        
        # Goal request subscriber
        self.goal_request_subscriber = self.create_subscription(
            String, '/navigation/goal_request', self.goal_request_callback, 10,
            callback_group=self.callback_group
        )
        
        # Navigation command subscriber
        self.nav_command_subscriber = self.create_subscription(
            PoseStamped, '/navigation/go_to_pose', self.nav_command_callback, 10,
            callback_group=self.callback_group
        )
    
    def _init_publishers(self):
        """Initialize ROS publishers"""
        self.joy_publisher = self.create_publisher(Joy, '/cerebri/in/joy', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Status publishers
        self.nav_status_publisher = self.create_publisher(
            String, '/navigation/status', 10
        )
        self.nav_feedback_publisher = self.create_publisher(
            String, '/navigation/feedback', 10
        )
        self.goal_reached_publisher = self.create_publisher(
            Bool, '/navigation/goal_reached', 10
        )
    
    def _init_action_clients(self):
        """Initialize action clients"""
        self.nav_action_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.callback_group
        )
        
        # Wait for action server
        self.get_logger().info("Waiting for Nav2 action server...")
        if not self.nav_action_client.wait_for_server(timeout_sec=self.action_server_timeout):
            self.get_logger().error("Nav2 action server not available!")
        else:
            self.get_logger().info("Nav2 action server connected")
    
    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """Handle robot pose updates"""
        self.robot_pose = PoseStamped()
        self.robot_pose.header = msg.header
        self.robot_pose.pose = msg.pose.pose
    
    def map_callback(self, msg: OccupancyGrid):
        """Handle map updates"""
        self.global_map = msg
    
    def global_costmap_callback(self, msg: OccupancyGrid):
        """Handle global costmap updates"""
        self.global_costmap = msg
    
    def status_callback(self, msg: Status):
        """Handle robot status updates"""
        self.robot_armed = msg.armed
        self.robot_mode = msg.mode
        
        # Auto-arm robot if not armed
        if not self.robot_armed:
            self.arm_robot()
    
    def goal_request_callback(self, msg: String):
        """Handle goal requests from other nodes"""
        request = msg.data
        
        if request == "explore":
            self.start_exploration()
        elif request == "stop":
            self.stop_navigation()
        elif request == "recover":
            self.start_recovery()
        elif request.startswith("heuristic:"):
            # Parse heuristic navigation: "heuristic:angle:distance"
            parts = request.split(":")
            if len(parts) >= 3:
                try:
                    angle = float(parts[1])
                    distance = float(parts[2])
                    self.navigate_using_heuristic(angle, distance)
                except ValueError:
                    self.get_logger().error(f"Invalid heuristic command: {request}")
    
    def nav_command_callback(self, msg: PoseStamped):
        """Handle direct navigation commands"""
        goal = NavigationGoal(
            pose=msg,
            goal_type="direct_command",
            priority=10  # High priority
        )
        self.add_goal(goal)
    
    def control_loop(self):
        """Main control loop"""
        with self.navigation_lock:
            # Process goal queue
            if self.current_goal is None and self.goal_queue:
                self.current_goal = self.goal_queue.pop(0)
                self._execute_goal(self.current_goal)
            
            # Check for goal timeout
            if self.current_goal and hasattr(self.current_goal, 'start_time'):
                if time.time() - self.current_goal.start_time > self.current_goal.timeout:
                    self.get_logger().warn(f"Goal timeout: {self.current_goal.goal_type}")
                    self._handle_goal_failure("timeout")
            
            # Publish status
            self._publish_status()
    
    def add_goal(self, goal: NavigationGoal):
        """Add a navigation goal to the queue"""
        with self.navigation_lock:
            # Insert based on priority
            inserted = False
            for i, existing_goal in enumerate(self.goal_queue):
                if goal.priority > existing_goal.priority:
                    self.goal_queue.insert(i, goal)
                    inserted = True
                    break
            
            if not inserted:
                self.goal_queue.append(goal)
        
        self.get_logger().info(f"Added goal: {goal.goal_type} (priority: {goal.priority})")
    
    def _execute_goal(self, goal: NavigationGoal):
        """Execute a navigation goal"""
        if not self.robot_pose:
            self.get_logger().error("No robot pose available")

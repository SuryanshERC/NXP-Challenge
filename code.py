#!/usr/bin/env python3

# This script provides a complete state machine framework for the challenge.
# It handles navigation, QR decoding, object processing, and data submission.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile
from rclpy.duration import Duration

from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage, Joy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from nav2_msgs.action import NavigateToPose
from synapse_msgs.msg import WarehouseShelf, Status

import cv2
from pyzbar.pyzbar import decode
import numpy as np
import math
import yaml
from enum import Enum

# --- Participant Configuration ---
PROGRESS_TABLE_GUI = False # Set to False for submission

class ChallengeState(Enum):
    """
    Enum to manage the robot's state throughout the challenge.
    """
    IDLE = 0
    ARMING = 1
    # Step 1: Find the first shelf using the map
    LOCATING_FIRST_SHELF = 2
    # Step 2: Navigate to the side of the shelf for QR code
    NAVIGATING_TO_QR = 3
    # Step 3: Decode the QR code
    DECODING_QR = 4
    # Step 4: Navigate to the front of the shelf for object recognition
    NAVIGATING_TO_SHELF_FRONT = 5
    # Step 5: Process detected objects from YOLO node
    PROCESSING_OBJECTS = 6
    # Step 6: Publish data and wait for curtain reveal
    PUBLISHING_DATA = 7
    # Step 7: Calculate next shelf location and repeat
    CALCULATING_NEXT_GOAL = 8
    # End of challenge
    COMPLETED = 9
    # Recovery state
    RECOVERY = 10


class Explore(Node):
    def __init__(self):
        super().__init__('explore')

        # --- Parameters ---
        self.declare_parameter('warehouse_id', 1)
        self.declare_parameter('shelf_count', 2)
        self.declare_parameter('initial_angle', 135.0)
        self.warehouse_id = self.get_parameter('warehouse_id').get_parameter_value().integer_value
        self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value
        self.get_logger().info(f"Warehouse ID: {self.warehouse_id}, Shelf Count: {self.shelf_count}, Initial Angle: {self.initial_angle}")

        # --- State Machine ---
        self.state = ChallengeState.IDLE
        self.target_shelf_id = 1
        self.shelf_data = {} # Dictionary to store data for all shelves

        # --- ROS 2 Communications ---
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.camera_sub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, 10)
        self.shelf_objects_sub = self.create_subscription(WarehouseShelf, '/shelf_objects', self.shelf_objects_callback, 10)
        self.status_sub = self.create_subscription(Status, '/cerebri/out/status', self.status_callback, 1)

        self.publisher_shelf_data = self.create_publisher(WarehouseShelf, '/shelf_data', 10)
        self.joy_pub = self.create_publisher(Joy, '/cerebri/in/joy', 1)
        self.qr_debug_pub = self.create_publisher(CompressedImage, '/debug_images/qr_code', 10)

        # --- Data Storage ---
        self.robot_pose = None
        self.qr_code_str = None
        self.shelf_objects_curr = None
        self.nav_goal_handle = None
        self.is_armed = False

        # --- Main Logic Timer ---
        self.timer = self.create_timer(1.0, self.main_loop)
        self.get_logger().info("Explore node initialized. Starting main loop.")

    def main_loop(self):
        """Main state machine loop."""
        if self.state == ChallengeState.IDLE:
            self.get_logger().info("State: IDLE. Attempting to arm robot.")
            self.state = ChallengeState.ARMING

        elif self.state == ChallengeState.ARMING:
            if not self.is_armed:
                self.arm_robot()
            else:
                self.get_logger().info("Robot is armed. Locating first shelf.")
                self.state = ChallengeState.LOCATING_FIRST_SHELF

        elif self.state == ChallengeState.LOCATING_FIRST_SHELF:
            # For simplicity, we use the initial angle and a fixed distance to find the first shelf.
            # A more robust solution would analyze the costmap.
            self.get_logger().info("State: LOCATING_FIRST_SHELF")
            first_shelf_pose = self.get_next_goal_from_heuristic(self.robot_pose, self.initial_angle, 4.0)
            self.shelf_data[self.target_shelf_id] = {'pose': first_shelf_pose}
            self.go_to_pose(first_shelf_pose, "side")
            self.state = ChallengeState.NAVIGATING_TO_QR
            
        elif self.state == ChallengeState.NAVIGATING_TO_QR:
            if self.is_goal_done():
                self.get_logger().info(f"Arrived at Shelf {self.target_shelf_id} QR location. Starting decoding.")
                self.state = ChallengeState.DECODING_QR
        
        elif self.state == ChallengeState.DECODING_QR:
            if self.qr_code_str:
                self.get_logger().info(f"QR Decoded: {self.qr_code_str}")
                self.shelf_data[self.target_shelf_id]['qr'] = self.qr_code_str
                shelf_pose = self.shelf_data[self.target_shelf_id]['pose']
                self.go_to_pose(shelf_pose, "front")
                self.state = ChallengeState.NAVIGATING_TO_SHELF_FRONT
                self.qr_code_str = None # Reset for next shelf
        
        elif self.state == ChallengeState.NAVIGATING_TO_SHELF_FRONT:
            if self.is_goal_done():
                self.get_logger().info(f"Arrived at Shelf {self.target_shelf_id} front. Looking for objects.")
                self.state = ChallengeState.PROCESSING_OBJECTS

        elif self.state == ChallengeState.PROCESSING_OBJECTS:
            if self.shelf_objects_curr:
                self.get_logger().info("Objects detected. Publishing data.")
                self.shelf_data[self.target_shelf_id]['objects'] = self.shelf_objects_curr
                self.state = ChallengeState.PUBLISHING_DATA

        elif self.state == ChallengeState.PUBLISHING_DATA:
            self.publish_shelf_data_for_scoring(
                self.target_shelf_id,
                self.shelf_data[self.target_shelf_id]['qr'],
                self.shelf_data[self.target_shelf_id]['objects']
            )
            self.shelf_objects_curr = None # Reset for next shelf
            
            if self.target_shelf_id >= self.shelf_count:
                self.get_logger().info("All shelves processed. Challenge complete!")
                self.state = ChallengeState.COMPLETED
            else:
                self.state = ChallengeState.CALCULATING_NEXT_GOAL

        elif self.state == ChallengeState.CALCULATING_NEXT_GOAL:
            # Extract heuristic and calculate next goal
            qr_string = self.shelf_data[self.target_shelf_id]['qr']
            try:
                parts = qr_string.split('_')
                angle = float(parts[1])
                self.get_logger().info(f"Heuristic angle for next shelf: {angle}")
                # Use heuristic to find next shelf
                next_shelf_pose = self.get_next_goal_from_heuristic(self.robot_pose, angle, 6.0) # Assume 6m distance
                
                self.target_shelf_id += 1
                self.shelf_data[self.target_shelf_id] = {'pose': next_shelf_pose}
                self.go_to_pose(next_shelf_pose, "side")
                self.state = ChallengeState.NAVIGATING_TO_QR

            except (IndexError, ValueError) as e:
                self.get_logger().error(f"Could not parse QR string '{qr_string}': {e}. Entering RECOVERY.")
                self.state = ChallengeState.RECOVERY

        elif self.state == ChallengeState.COMPLETED:
            self.get_logger().info("Challenge finished. Stopping timer.")
            self.timer.cancel()

        elif self.state == ChallengeState.RECOVERY:
            self.get_logger().error("In recovery mode. Implement recovery logic here (e.g., explore randomly).")
            # For now, just stop
            self.state = ChallengeState.COMPLETED
            
    # --- Callback Functions ---
    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def status_callback(self, msg):
        self.is_armed = msg.armed

    def shelf_objects_callback(self, msg):
        self.shelf_objects_curr = msg
        detected = [f"{q}x {o}" for o, q in zip(msg.objects, msg.object_quantities)]
        if detected:
            self.get_logger().info(f"Objects seen: {', '.join(detected)}", throttle_duration_sec=5)

    def camera_image_callback(self, msg):
        """Callback to decode QR codes from the camera stream."""
        # Only process images if we are in the decoding state
        if self.state != ChallengeState.DECODING_QR:
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None: return

            decoded_objects = decode(cv_image)
            if not decoded_objects:
                self.qr_debug_pub.publish(msg) # Publish raw image if no QR found
                return

            for obj in decoded_objects:
                self.qr_code_str = obj.data.decode('utf-8')
                # Draw bounding box for debugging
                points = obj.polygon
                cv2.polylines(cv_image, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
                break 

            # Publish the debug image with the bounding box
            _, compressed_jpeg = cv2.imencode('.jpg', cv_image)
            debug_msg = CompressedImage()
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.format = "jpeg"
            debug_msg.data = compressed_jpeg.tobytes()
            self.qr_debug_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'Error in QR code callback: {e}')
            
    # --- Action and Navigation ---
    def arm_robot(self):
        joy_msg = Joy()
        joy_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # Corresponds to arming button
        self.joy_pub.publish(joy_msg)
        self.get_logger().info("Sent arming command.", throttle_duration_sec=5)

    def go_to_pose(self, pose, target_type="front"):
        """Sends a navigation goal to Nav2."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        
        q = self.euler_to_quaternion(0, 0, pose['yaw'])
        
        # Adjust position based on whether we are viewing QR code or front
        # This is a simplification. You might need to adjust these offsets.
        if target_type == "side":
            # Position the robot to the side of the shelf
            offset_x = 0.8 * math.cos(pose['yaw'] + math.pi/2)
            offset_y = 0.8 * math.sin(pose['yaw'] + math.pi/2)
        else: # "front"
            # Position robot in front of the shelf
            offset_x = -1.5 * math.cos(pose['yaw'])
            offset_y = -1.5 * math.sin(pose['yaw'])
        
        goal_msg.pose.pose.position.x = pose['x'] + offset_x
        goal_msg.pose.pose.position.y = pose['y'] + offset_y
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        self.nav_to_pose_client.wait_for_server()
        self.get_logger().info(f"Sending goal to {target_type} of shelf: ({goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f})")
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg, feedback_callback=self.nav_feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.nav_goal_handle = future.result()
        if not self.nav_goal_handle.accepted:
            self.get_logger().error('Goal rejected by server')
            return
        result_future = self.nav_goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        # You can add logic here to monitor navigation progress
        pass

    def goal_result_callback(self, future):
        status = future.result().status
        if status != 4: # 4 is SUCCEEDED
             self.get_logger().warn(f'Navigation goal failed with status code: {status}. Consider recovery.')
             # self.state = ChallengeState.RECOVERY

    def is_goal_done(self):
        """Check if the current navigation goal is complete."""
        if not self.nav_goal_handle:
            return False
        return self.nav_goal_handle.get_status() in [4, 5, 6] # SUCCEEDED, CANCELED, ABORTED

    # --- Helper Functions ---
    def get_next_goal_from_heuristic(self, current_pose, angle_deg, distance_m):
        """Calculates a new goal pose based on current position, an angle, and a distance."""
        if not current_pose:
            self.get_logger().warn("Cannot calculate next goal, robot pose is unknown.")
            return {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

        angle_rad = math.radians(angle_deg)
        current_yaw = self.quaternion_to_euler(current_pose.orientation)[2]
        
        # Heuristic angle is from the world x-axis.
        # Robot's starting pose is the world frame origin.
        new_x = current_pose.position.x + distance_m * math.cos(angle_rad)
        new_y = current_pose.position.y + distance_m * math.sin(angle_rad)
        
        # Point the robot towards the new location from the old one
        new_yaw = math.atan2(new_y - current_pose.position.y, new_x - current_pose.position.x)
        
        return {'x': new_x, 'y': new_y, 'yaw': new_yaw}

    def publish_shelf_data_for_scoring(self, shelf_id, qr_string, detected_objects_msg):
        """Constructs and publishes the final message for scoring."""
        msg = WarehouseShelf()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.warehouse_id = self.warehouse_id
        msg.shelf_id = shelf_id
        msg.qr_code = qr_string
        if detected_objects_msg:
            msg.objects = detected_objects_msg.objects
            msg.object_quantities = detected_objects_msg.object_quantities
        
        self.publisher_shelf_data.publish(msg)
        self.get_logger().info(f"Published final data for Shelf ID {shelf_id}.")

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        q = [0.0]*4
        q[0] = cy * cp * sr - sy * sp * cr  # x
        q[1] = sy * cp * sr + cy * sp * cr  # y
        q[2] = sy * cp * cr - cy * sp * sr  # z
        q[3] = cy * cp * cr + sy * sp * sr  # w
        return q

    def quaternion_to_euler(self, q):
        t0 = +2.0 * (q.w * q.x + q.y * q.z)
        t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (q.w * q.y - q.z * q.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_z = math.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z


def main(args=None):
    rclpy.init(args=args)
    node = Explore()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

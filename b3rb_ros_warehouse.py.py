# Copyright 2025 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import threading

# Import for QR Code decoding
from pyzbar import pyzbar

from sensor_msgs.msg import Joy, CompressedImage
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from synapse_msgs.msg import Status, WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0
PROGRESS_TABLE_GUI = True # Set to False for final submission

class WindowProgressTable:
    """Optional GUI to visualize challenge progress. Unchanged from original."""
    def __init__(self, root, shelf_count):
        self.root = root
        self.root.title("Warehouse Challenge Progress")
        self.root.attributes("-topmost", True)
        self.row_count = 2
        self.col_count = shelf_count
        self.boxes = []
        for row in range(self.row_count):
            row_boxes = []
            for col in range(self.col_count):
                box = tk.Text(root, width=15, height=4, wrap=tk.WORD, borderwidth=1,
                              relief="solid", font=("Helvetica", 12))
                box.insert(tk.END, f"Shelf {col+1}")
                box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                row_boxes.append(box)
            self.boxes.append(row_boxes)
        for row in range(self.row_count): self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.col_count): self.root.grid_columnconfigure(col, weight=1)
    def change_box_color(self, row, col, color): self.boxes[row][col].config(bg=color)
    def change_box_text(self, row, col, text):
        self.boxes[row][col].delete(1.0, tk.END); self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
    global box_app
    root = tk.Tk()
    box_app = WindowProgressTable(root, shelf_count)
    root.mainloop()

class WarehouseExplore(Node):
    """Main node to manage the Warehouse Treasure Hunt challenge logic."""
    def __init__(self):
        super().__init__('warehouse_explore')

        # --- Action & Service Clients ---
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # --- Publishers & Subscribers ---
        self.subscription_pose = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, QOS_PROFILE_DEFAULT)
        self.subscription_global_map = self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.global_map_callback, QOS_PROFILE_DEFAULT)
        self.subscription_simple_map = self.create_subscription(OccupancyGrid, '/map', self.simple_map_callback, QOS_PROFILE_DEFAULT)
        self.subscription_status = self.create_subscription(Status, '/cerebri/out/status', self.cerebri_status_callback, QOS_PROFILE_DEFAULT)
        self.subscription_shelf_objects = self.create_subscription(WarehouseShelf, '/shelf_objects', self.shelf_objects_callback, QOS_PROFILE_DEFAULT)
        self.subscription_camera = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.camera_image_callback, QOS_PROFILE_DEFAULT)
        
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_qr_decode = self.create_publisher(CompressedImage, "/debug_images/qr_code", QOS_PROFILE_DEFAULT)
        self.publisher_shelf_data = self.create_publisher(WarehouseShelf, "/shelf_data", QOS_PROFILE_DEFAULT)

        # --- Parameters ---
        self.declare_parameter('shelf_count', 1)
        self.declare_parameter('initial_angle', 0.0)
        self.shelf_count = self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = self.get_parameter('initial_angle').get_parameter_value().double_value

        # --- Robot & System State ---
        self.armed = False
        self.logger = self.get_logger()
        self.pose_curr = PoseWithCovarianceStamped()
        self.buggy_pose_x, self.buggy_pose_y = 0.0, 0.0
        
        # --- Map Data ---
        self.simple_map_curr, self.global_map_curr = None, None

        # --- Goal Management & Recovery ---
        self.goal_completed = True
        self.goal_handle_curr = None
        self.cancelling_goal = False
        self.recovery_threshold = 10
        self._frame_id = "map"
        self.nav_retry_count = 0
        self.max_nav_retries = 3 # Max attempts to navigate to QR/Object spots before skipping

        # --- Challenge State Machine ---
        self.states = ["IDLE", "FINDING_SHELF", "NAVIGATING_TO_QR", "DECODING_QR", "NAVIGATING_TO_OBJECTS", "READING_OBJECTS", "PUBLISHING_DATA", "SKIPPING_SHELF", "COMPLETED"]
        self.current_state = "IDLE"

        # --- Challenge Data Management ---
        self.target_shelf_id = 1
        self.shelf_data = {i: {} for i in range(1, self.shelf_count + 1)}
        self.last_qr_decoded_data = None
        self.shelf_objects_buffer = None
        self.shelf_locations = {}

        # --- Main Control Loop ---
        self.main_timer = self.create_timer(1.0, self.main_control_loop)
        self.logger.info(f"Warehouse Explorer Initialized. State machine started.")
        
    def set_state(self, new_state):
        if new_state in self.states and self.current_state != new_state:
            self.logger.info(f"State changed from {self.current_state} -> {new_state}")
            self.current_state = new_state
            # Reset retry counter when starting a new navigation sequence
            if new_state in ["NAVIGATING_TO_QR", "NAVIGATING_TO_OBJECTS"]:
                self.nav_retry_count = 0

    # =========================================================================================
    # MAIN CONTROL LOOP - The heart of the challenge logic
    # =========================================================================================
    def main_control_loop(self):
        if not self.armed:
            self.logger.warn("Robot is not armed.", throttle_skip_first=True, throttle_duration_sec=5)
            return

        # State transition logic
        if self.current_state == "IDLE":
            if self.global_map_curr is not None: self.set_state("FINDING_SHELF")

        elif self.current_state == "FINDING_SHELF":
            if self.goal_completed:
                shelf_found, center, yaw = self.find_target_shelf_on_map(self.target_shelf_id)
                if shelf_found:
                    self.logger.info(f"Shelf {self.target_shelf_id} found at {center} with yaw {math.degrees(yaw):.1f} deg")
                    self.shelf_locations[self.target_shelf_id] = {'center': center, 'yaw': yaw}
                    self.set_state("NAVIGATING_TO_QR")
                else:
                    self.logger.info(f"Shelf {self.target_shelf_id} not visible. Exploring...")
                    self.explore_the_warehouse()

        elif self.current_state == "NAVIGATING_TO_QR":
            if self.goal_completed:
                shelf_info = self.shelf_locations.get(self.target_shelf_id)
                if shelf_info:
                    goal_pose = self.get_qr_scan_pose(shelf_info)
                    if self.send_goal_from_world_pose(goal_pose): self.set_state("DECODING_QR")
        
        elif self.current_state == "DECODING_QR":
            if self.goal_completed:
                self.nav_retry_count += 1
                self.logger.warn(f"Arrived at QR spot but no QR found. Retry attempt {self.nav_retry_count}/{self.max_nav_retries}.")
                if self.nav_retry_count >= self.max_nav_retries:
                    self.set_state("SKIPPING_SHELF")
                else:
                    self.set_state("NAVIGATING_TO_QR") # Try navigating again

        elif self.current_state == "NAVIGATING_TO_OBJECTS":
            if self.goal_completed:
                shelf_info = self.shelf_locations.get(self.target_shelf_id)
                if shelf_info:
                    goal_pose = self.get_object_scan_pose(shelf_info)
                    if self.send_goal_from_world_pose(goal_pose): self.set_state("READING_OBJECTS")
        
        elif self.current_state == "READING_OBJECTS":
            if self.goal_completed:
                self.nav_retry_count += 1
                self.logger.warn(f"Arrived at object spot but no objects read. Retry attempt {self.nav_retry_count}/{self.max_nav_retries}.")
                if self.nav_retry_count >= self.max_nav_retries:
                    self.set_state("SKIPPING_SHELF")
                else:
                    self.set_state("NAVIGATING_TO_OBJECTS")

        elif self.current_state == "PUBLISHING_DATA":
            self.publish_shelf_result(self.target_shelf_id)
            if self.target_shelf_id == self.shelf_count:
                self.set_state("COMPLETED")
            else:
                self.target_shelf_id += 1
                self.set_state("FINDING_SHELF")

        elif self.current_state == "SKIPPING_SHELF":
            self.logger.error(f"Failed to process Shelf {self.target_shelf_id} after multiple retries. Skipping.")
            if self.target_shelf_id == self.shelf_count:
                self.set_state("COMPLETED")
            else:
                self.target_shelf_id += 1
                self.set_state("FINDING_SHELF")

        elif self.current_state == "COMPLETED":
            self.logger.info("All shelves processed. Task complete!")
            self.main_timer.cancel()

    # =========================================================================================
    # CORE CHALLENGE IMPLEMENTATION - TUNE PARAMETERS HERE
    # =========================================================================================
    def find_target_shelf_on_map(self, shelf_id):
        if self.simple_map_curr is None: self.logger.warn("Map data not available."); return False, None, None
        
        map_data = self.simple_map_curr
        map_info, map_array = map_data.info, np.array(map_data.data).reshape((map_data.info.height, map_data.info.width))
        binary_map = (map_array == 100).astype(np.uint8)
        labeled_map, num_features = label(binary_map)
        
        # *** CRITICAL TUNING PARAMETERS ***
        min_shelf_pixels = 40   # Tune based on map resolution and shelf size
        max_shelf_pixels = 600  # Tune this
        search_distance_meters = 7.0 # Tune based on warehouse layout
        
        potential_shelves = []
        for i in range(1, num_features + 1):
            pixel_count = np.sum(labeled_map == i)
            if min_shelf_pixels <= pixel_count <= max_shelf_pixels:
                center_map_yx = center_of_mass(binary_map, labeled_map, i)
                potential_shelves.append({'label': i, 'center_yx': center_map_yx})
        
        if not potential_shelves: self.logger.info("No obstacles match shelf size criteria."); return False, None, None

        target_blob = None
        if shelf_id == 1:
            if not self.shelf_locations: target_blob = potential_shelves[0]
            else: # If we already found shelves, find the one not yet in our list
                known_labels = [loc.get('label') for loc in self.shelf_locations.values()]
                for p_shelf in potential_shelves:
                    if p_shelf['label'] not in known_labels: target_blob = p_shelf; break
        else:
            prev_shelf_id = shelf_id - 1
            if prev_shelf_id not in self.shelf_locations or 'qr_data' not in self.shelf_data[prev_shelf_id]:
                self.logger.error(f"Cannot find Shelf {shelf_id}: Missing data for previous shelf {prev_shelf_id}."); return False, None, None
            
            prev_center_world = self.shelf_locations[prev_shelf_id]['center']
            heuristic_angle_rad = math.radians(self.shelf_data[prev_shelf_id]['qr_data']['angle'])
            expected_world_x = prev_center_world[0] + search_distance_meters * math.cos(heuristic_angle_rad)
            expected_world_y = prev_center_world[1] + search_distance_meters * math.sin(heuristic_angle_rad)
            
            min_dist, closest_shelf = float('inf'), None
            for shelf_blob in potential_shelves:
                center_world_x, center_world_y = self.get_world_coord_from_map_coord(shelf_blob['center_yx'][1], shelf_blob['center_yx'][0], map_info)
                dist = euclidean((expected_world_x, expected_world_y), (center_world_x, center_world_y))
                if dist < min_dist: min_dist, closest_shelf = dist, shelf_blob
            target_blob = closest_shelf
        
        if target_blob is None: self.logger.warn(f"Could not identify target blob for Shelf {shelf_id}."); return False, None, None

        target_label = target_blob['label']
        shelf_pixels_yx = np.argwhere(labeled_map == target_label)
        pca = PCA(n_components=2); pca.fit(shelf_pixels_yx)
        orientation_rad = math.atan2(pca.components_[0][0], pca.components_[0][1])
        center_world_xy = self.get_world_coord_from_map_coord(target_blob['center_yx'][1], target_blob['center_yx'][0], map_info)
        
        # Store label to avoid finding the same shelf again
        self.shelf_locations[shelf_id] = {**self.shelf_locations.get(shelf_id, {}), 'label': target_label}
        
        return True, center_world_xy, orientation_rad

    def get_qr_scan_pose(self, shelf_info):
        center, yaw = shelf_info['center'], shelf_info['yaw']
        offset_dist = 1.6  # meters away from the side
        side_angle = yaw + math.pi / 2.0
        
        goal_x = center[0] - offset_dist * math.sin(yaw)
        goal_y = center[1] + offset_dist * math.cos(yaw)
        goal_yaw = self.create_yaw_from_vector(center[0], center[1], goal_x, goal_y)
        
        self.logger.info(f"Calculated QR scan pose for shelf.")
        return self.create_goal_from_world_coord(goal_x, goal_y, goal_yaw)

    def get_object_scan_pose(self, shelf_info):
        center, yaw = shelf_info['center'], shelf_info['yaw']
        offset_dist = 2.0  # meters away from the front
        
        goal_x = center[0] - offset_dist * math.cos(yaw)
        goal_y = center[1] - offset_dist * math.sin(yaw)
        goal_yaw = self.create_yaw_from_vector(center[0], center[1], goal_x, goal_y)
        
        self.logger.info(f"Calculated object scan pose for shelf.")
        return self.create_goal_from_world_coord(goal_x, goal_y, goal_yaw)

    def explore_the_warehouse(self):
        if self.global_map_curr is None or not self.goal_completed: return
        map_array = np.array(self.global_map_curr.data).reshape((self.global_map_curr.info.height, self.global_map_curr.info.width))
        frontiers = self.get_frontiers_for_space_exploration(map_array)
        if frontiers:
            buggy_map_y, buggy_map_x = self.get_map_coord_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, self.global_map_curr.info)
            frontiers.sort(key=lambda p: euclidean(p, (buggy_map_y, buggy_map_x)))
            fy, fx = frontiers[0]
            goal = self.create_goal_from_map_coord(fx, fy, self.global_map_curr.info)
            self.send_goal_from_world_pose(goal)

    # =========================================================================================
    # CALLBACKS AND DATA HANDLING
    # =========================================================================================
    def camera_image_callback(self, message):
        if self.current_state != "DECODING_QR": return
        np_arr, image = np.frombuffer(message.data, np.uint8), cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        barcodes = pyzbar.decode(image)
        for barcode in barcodes:
            qr_string = barcode.data.decode("utf-8")
            try:
                shelf_id, angle_str, _ = qr_string.split('_')
                if int(shelf_id) == self.target_shelf_id:
                    self.shelf_data[self.target_shelf_id]['qr_data'] = {'id':int(shelf_id), 'angle':float(angle_str), 'raw':qr_string}
                    self.logger.info(f"Successfully decoded QR for target shelf {self.target_shelf_id}!")
                    self.cancel_current_goal()
                    self.set_state("NAVIGATING_TO_OBJECTS")
                    break
            except (ValueError, IndexError): self.logger.warn(f"Decoded malformed QR: {qr_string}")
        self.publish_debug_image(self.publisher_qr_decode, image)

    def shelf_objects_callback(self, message):
        if self.current_state != "READING_OBJECTS": return
        self.shelf_data[self.target_shelf_id]['objects'] = message
        self.logger.info(f"Object data received for shelf {self.target_shelf_id}: {message.object_name}")
        self.cancel_current_goal()
        self.set_state("PUBLISHING_DATA")

    def publish_shelf_result(self, shelf_id):
        shelf_info = self.shelf_data[shelf_id]
        if 'qr_data' not in shelf_info or 'objects' not in shelf_info:
            self.logger.error(f"Cannot publish for shelf {shelf_id}: Missing data."); return

        msg = WarehouseShelf()
        msg.object_name = shelf_info['objects'].object_name
        msg.object_count = shelf_info['objects'].object_count
        msg.qr_decoded = shelf_info['qr_data']['raw']
        self.publisher_shelf_data.publish(msg)
        self.logger.info(f"Published final data for shelf {shelf_id}.")

        if PROGRESS_TABLE_GUI and box_app:
            obj_str = "\n".join([f"{n}:{c}" for n,c in zip(msg.object_name,msg.object_count)])
            box_app.change_box_text(0, shelf_id-1, obj_str); box_app.change_box_color(0, shelf_id-1, "lightgreen")
            box_app.change_box_text(1, shelf_id-1, msg.qr_decoded); box_app.change_box_color(1, shelf_id-1, "lightblue")
            
    # =========================================================================================
    # Unchanged Helper and ROS Communication Functions
    # =========================================================================================
    def pose_callback(self, message): self.pose_curr=message; self.buggy_pose_x=message.pose.pose.position.x; self.buggy_pose_y=message.pose.pose.position.y
    def simple_map_callback(self, message): self.simple_map_curr=message
    def global_map_callback(self, message): self.global_map_curr=message
    def cerebri_status_callback(self, message):
        if not self.armed and message.mode==3 and message.arming==2: self.armed=True; self.logger.info("Robot is armed.")
        elif self.armed: return
        else: msg=Joy(); msg.buttons=[0,1,0,0,0,0,0,1]; self.publisher_joy.publish(msg)
    def publish_debug_image(self,p,i): msg=CompressedImage();_,d=cv2.imencode('.jpg',i);msg.format="jpeg";msg.data=d.tobytes();p.publish(msg)
    def get_frontiers_for_space_exploration(self,m):
        f=[];_,nf=label(m==-1)
        for i in range(1,nf+1):
            c=np.argwhere(label(m==-1)==i)
            for r,c_ in c:
                if 0<r<m.shape[0]-1 and 0<c_<m.shape[1]-1 and np.any(m[r-1:r+2,c_-1:c_+2]==0): f.extend(list(c)); break
        return f
    def cancel_current_goal(self):
        if self.goal_handle_curr and self.goal_handle_curr.status==GoalStatus.STATUS_EXECUTING and not self.cancelling_goal:
            self.cancelling_goal=True;self.logger.info("Cancelling goal...");f=self.action_client._cancel_goal_async(self.goal_handle_curr);f.add_done_callback(self.cancel_goal_callback)
    def cancel_goal_callback(self,f): self.cancelling_goal=False;self.logger.info("Goal cancel request processed.")
    def goal_result_callback(self,f):
        s=f.result().status
        if s!=GoalStatus.STATUS_SUCCEEDED: self.logger.warn(f"Goal failed: {s}")
        self.goal_completed=True;self.goal_handle_curr=None
    def goal_response_callback(self,f):
        h=f.result()
        if not h.accepted:self.logger.warn('Goal rejected.');self.goal_completed=True
        else:self.goal_handle_curr=h;f=h.get_result_async();f.add_done_callback(self.goal_result_callback)
    def goal_feedback_callback(self,m):
        if m.feedback.number_of_recoveries>self.recovery_threshold:self.cancel_current_goal()
    def send_goal_from_world_pose(self,p):
        if not self.goal_completed:return False
        self.goal_completed=False
        if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):self.logger.error('Action server unavailable!');return False
        g=NavigateToPose.Goal();g.pose=p;f=self.action_client.send_goal_async(g,self.goal_feedback_callback);f.add_done_callback(self.goal_response_callback);return True
    def _gmi(self,mi): return(mi.resolution,mi.origin.position.x,mi.origin.position.y) if mi else None
    def get_world_coord_from_map_coord(self,mx,my,mi):
        i=self._gmi(mi)
        return((mx+0.5)*i[0]+i[1],(my+0.5)*i[0]+i[2]) if i else(0.,0.)
    def get_map_coord_from_world_coord(self,wx,wy,mi):
        i=self._gmi(mi)
        return(int((wx-i[1])/i[0]),int((wy-i[2])/i[0])) if i and i[0]>0 else(0,0)
    def _c qfy(self,y):q=Quaternion();q.z=math.sin(y*0.5);q.w=math.cos(y*0.5);return q
    def create_yaw_from_vector(self,dx,dy,sx,sy):return math.atan2(dy-sy,dx-sx)
    def create_goal_from_world_coord(self,wx,wy,y=None):
        g=PoseStamped();g.header.stamp=self.get_clock().now().to_msg();g.header.frame_id=self._frame_id
        g.pose.position.x=wx;g.pose.position.y=wy
        if y is None:y=self.create_yaw_from_vector(wx,wy,self.buggy_pose_x,self.buggy_pose_y)
        g.pose.orientation=self._c qfy(y);return g
    def create_goal_from_map_coord(self,mx,my,mi,y=None):wx,wy=self.get_world_coord_from_map_coord(mx,my,mi);return self.create_goal_from_world_coord(wx,wy,y)

def main(args=None):
    rclpy.init(args=args)
    node=WarehouseExplore()
    if PROGRESS_TABLE_GUI:threading.Thread(target=run_gui,args=(node.shelf_count,),daemon=True).start()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# Copyright 2025 NXP
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
import numpy as np
import cv2
import yaml
from ament_index_python.packages import get_package_share_directory

# TFLite-runtime is used for inference as per the project description
import tflite_runtime.interpreter as tflite

from sensor_msgs.msg import CompressedImage
from synapse_msgs.msg import WarehouseShelf

class ObjectRecognitionNode(Node):
    """Node for running YOLOv5 TFLite model for object recognition."""
    def __init__(self):
        super().__init__('object_recognition_node')

        # --- Parameters & Paths ---
        # Get paths to model and config files from the package share directory
        package_share_dir = get_package_share_directory('b3rb_ros_aim_india')
        model_path = f"{package_share_dir}/resource/yolov5n-int8.tflite"
        coco_yaml_path = f"{package_share_dir}/resource/coco.yaml"
        
        # --- Publishers & Subscribers ---
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10)
        self.publisher_shelf_objects = self.create_publisher(
            WarehouseShelf,
            '/shelf_objects',
            10)
        self.publisher_debug_image = self.create_publisher(
            CompressedImage,
            '/debug_images/object_recog',
            10)

        # --- Load YOLOv5 TFLite Model ---
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.img_height = self.input_details[0]['shape'][1]
            self.img_width = self.input_details[0]['shape'][2]
            self.logger.info(f"TFLite model loaded successfully from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load TFLite model: {e}")
            return

        # --- Load Class Names from COCO YAML ---
        try:
            with open(coco_yaml_path, 'r') as f:
                coco_data = yaml.safe_load(f)
                self.class_names = coco_data['names']
            self.logger.info(f"Loaded {len(self.class_names)} class names from {coco_yaml_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load coco.yaml: {e}")
            self.class_names = [] # Fallback

        self.get_logger().info("Object Recognition Node has been initialized.")

    def image_callback(self, msg):
        """Processes an incoming image, runs inference, and publishes results."""
        try:
            # 1. Decode and Preprocess Image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            input_image = self.preprocess_image(image_bgr)

            # 2. Run Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()
            detections = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # 3. Post-process Detections and Publish
            object_counts, drawn_image = self.postprocess_and_draw(detections, image_bgr)
            
            if object_counts:
                shelf_msg = WarehouseShelf()
                shelf_msg.object_name = list(object_counts.keys())
                shelf_msg.object_count = list(object_counts.values())
                self.publisher_shelf_objects.publish(shelf_msg)

            # 4. Publish Debug Image
            debug_msg = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', drawn_image)
            debug_msg.format = "jpeg"
            debug_msg.data = encoded_data.tobytes()
            self.publisher_debug_image.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def preprocess_image(self, image_bgr):
        """Resizes and normalizes the image for the TFLite model."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image_rgb, (self.img_width, self.img_height))
        # Normalize to [0, 1] and add batch dimension
        input_data = np.expand_dims(resized_image, axis=0).astype(np.float32) / 255.0
        return input_data

    def postprocess_and_draw(self, detections, original_image, conf_threshold=0.45, nms_threshold=0.5):
        """Applies NMS, counts objects, and draws bounding boxes."""
        original_h, original_w, _ = original_image.shape
        boxes, confidences, class_ids = [], [], []

        for detection in detections:
            # The output format is [x_center, y_center, width, height, confidence, class_scores...]
            confidence = detection[4]
            if confidence > conf_threshold:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                
                # Convert bbox from model's scale to original image scale
                x_center, y_center, w, h = detection[:4]
                x1 = int((x_center - w/2) * original_w)
                y1 = int((y_center - h/2) * original_h)
                x2 = int((x_center + w/2) * original_w)
                y2 = int((y_center + h/2) * original_h)

                boxes.append([x1, y1, x2-x1, y2-y1]) # cv2.NMSBoxes expects (x, y, w, h)
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        object_counts = {}
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                class_id = class_ids[i]
                class_name = self.class_names[class_id]
                
                # Count objects
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Draw on image
                color = (0, 255, 0)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)
                label = f"{class_name}: {confidences[i]:.2f}"
                cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return object_counts, original_image

def main(args=None):
    rclpy.init(args=args)
    object_recognition_node = ObjectRecognitionNode()
    rclpy.spin(object_recognition_node)
    object_recognition_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
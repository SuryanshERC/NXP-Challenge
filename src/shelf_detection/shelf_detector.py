import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from scipy.ndimage import label
from .map_processor import MapProcessor

class ShelfDetector(Node):
    def __init__(self):
        super().__init__('shelf_detector')
        self.get_logger().info('Starting Shelf Detector Node')

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.camera_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.camera_callback, 10
        )

        # Publishers
        self.shelf_pose_pub = self.create_publisher(PoseStamped, '/shelf_pose', 10)

        # Map processor
        self.map_processor = MapProcessor()

        # Parameters
        self.detection_method = 'map'  # Options: 'map' or 'image'
        self.shelf_size = 1.0  # Approximate shelf size in meters (adjust based on world)

    def map_callback(self, msg):
        """Process occupancy grid to detect shelves."""
        if self.detection_method != 'map':
            return

        # Convert occupancy grid to numpy array
        width, height = msg.info.width, msg.info.height
        data = np.array(msg.info.data, dtype=np.int8).reshape(height, width)

        # Detect shelves
        shelf_centers = self.map_processor.detect_shelves(data, msg.info.resolution)
        for center in shelf_centers:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = center[0] * msg.info.resolution + msg.info.origin.position.x
            pose.pose.position.y = center[1] * msg.info.resolution + msg.info.origin.position.y
            pose.pose.orientation.w = 1.0  # Default orientation (adjust if needed)
            self.shelf_pose_pub.publish(pose)
            self.get_logger().info(f'Detected shelf at: ({pose.pose.position.x}, {pose.pose.position.y})')

    def camera_callback(self, msg):
        """Process camera image for shelf detection (placeholder)."""
        if self.detection_method != 'image':
            return

        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Placeholder for image-based shelf detection
        # Implement contour detection or feature matching here
        self.get_logger().info('Image-based shelf detection not implemented')

def main(args=None):
    rclpy.init(args=args)
    node = ShelfDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

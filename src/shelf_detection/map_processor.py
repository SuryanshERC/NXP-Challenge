import numpy as np
from scipy.ndimage import label

class MapProcessor:
    def __init__(self):
        self.min_shelf_area = 50  # Minimum pixels for a shelf (adjust based on resolution)
        self.max_shelf_area = 200  # Maximum pixels for a shelf
        self.occupancy_threshold = 80  # Occupancy probability threshold (0-100)

    def detect_shelves(self, map_data, resolution):
        """
        Detect shelves in the occupancy grid.
        Args:
            map_data: 2D numpy array of occupancy probabilities
            resolution: Meters per grid cell
        Returns:
            List of (x, y) centers of detected shelves in grid coordinates
        """
        # Threshold the map to identify occupied regions
        occupied = (map_data >= self.occupancy_threshold).astype(np.uint8)

        # Label connected regions
        labeled_array, num_features = label(occupied)

        # Find centers of potential shelves
        shelf_centers = []
        for i in range(1, num_features + 1):
            region = (labeled_array == i)
            area = np.sum(region)
            if self.min_shelf_area <= area <= self.max_shelf_area:
                # Calculate center of mass
                y, x = np.where(region)
                center_x = np.mean(x)
                center_y = np.mean(y)
                shelf_centers.append((center_x, center_y))

        return shelf_centers

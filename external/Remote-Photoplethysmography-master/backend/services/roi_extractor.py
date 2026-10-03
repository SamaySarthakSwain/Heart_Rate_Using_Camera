"""ROI extraction service for stable skin regions."""

import cv2
import numpy as np


# MediaPipe FaceMesh landmark indices for skin ROIs
# These define polygons over stable skin areas (forehead, cheeks)
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 299, 337, 338, 297]
LEFT_CHEEK_LANDMARKS = [36, 50, 187, 205, 206, 216]
RIGHT_CHEEK_LANDMARKS = [266, 280, 411, 425, 426, 436]


class ROIExtractor:
    """Extracts mean RGB values from facial skin regions of interest."""

    def __init__(self):
        """Initialize ROI extractor."""
        self.roi_definitions = {
            "forehead": FOREHEAD_LANDMARKS,
            "left_cheek": LEFT_CHEEK_LANDMARKS,
            "right_cheek": RIGHT_CHEEK_LANDMARKS,
        }

    def extract(
        self, frame: np.ndarray, landmarks: list[tuple[int, int]]
    ) -> dict | None:
        """Extract mean RGB from skin ROIs.

        Args:
            frame: BGR image as numpy array.
            landmarks: List of (x, y) pixel coordinates from FaceMesh (468 points).

        Returns:
            Dict with 'rgb' (mean R, G, B), 'brightness' (mean intensity),
            'roi_pixels' (total pixel count), and per-ROI values.
            Returns None if extraction fails.
        """
        h, w = frame.shape[:2]
        all_r, all_g, all_b = [], [], []
        roi_details = {}
        total_pixels = 0

        for roi_name, indices in self.roi_definitions.items():
            try:
                # Get polygon points for this ROI
                points = np.array(
                    [landmarks[i] for i in indices if i < len(landmarks)],
                    dtype=np.int32,
                )

                if len(points) < 3:
                    continue

                # Create mask for this ROI polygon
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, points, 255)

                # Extract pixels within the ROI
                roi_pixels = frame[mask == 255]
                if len(roi_pixels) == 0:
                    continue

                # BGR → RGB mean values
                mean_b = float(np.mean(roi_pixels[:, 0]))
                mean_g = float(np.mean(roi_pixels[:, 1]))
                mean_r = float(np.mean(roi_pixels[:, 2]))

                all_r.append(mean_r)
                all_g.append(mean_g)
                all_b.append(mean_b)
                total_pixels += len(roi_pixels)

                roi_details[roi_name] = {
                    "r": mean_r,
                    "g": mean_g,
                    "b": mean_b,
                    "pixel_count": len(roi_pixels),
                }

            except (IndexError, ValueError):
                continue

        if not all_r:
            return None

        # Average across all ROIs
        mean_r = float(np.mean(all_r))
        mean_g = float(np.mean(all_g))
        mean_b = float(np.mean(all_b))
        brightness = float((mean_r + mean_g + mean_b) / 3.0)

        return {
            "rgb": (mean_r, mean_g, mean_b),
            "brightness": brightness,
            "roi_pixels": total_pixels,
            "roi_details": roi_details,
        }

    def draw_rois(
        self, frame: np.ndarray, landmarks: list[tuple[int, int]]
    ) -> np.ndarray:
        """Draw ROI polygons on frame for visualization.

        Args:
            frame: BGR image to draw on (will be modified in place).
            landmarks: List of (x, y) pixel coordinates.

        Returns:
            Frame with ROI polygons drawn.
        """
        colors = {
            "forehead": (0, 255, 0),     # Green
            "left_cheek": (255, 0, 0),    # Blue
            "right_cheek": (0, 0, 255),   # Red
        }

        for roi_name, indices in self.roi_definitions.items():
            points = np.array(
                [landmarks[i] for i in indices if i < len(landmarks)],
                dtype=np.int32,
            )
            if len(points) >= 3:
                cv2.polylines(frame, [points], True, colors[roi_name], 2)

        return frame

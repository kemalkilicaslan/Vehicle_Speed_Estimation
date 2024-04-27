# Vehicle Speed Estimation

# Import the necessary libraries
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque

# Defines an image transformer class with ROI_COORDINATES and target coordinates
# Varies according to video size
ROI_COORDINATES = np.array([[1290, 747], [2324, 747], [3431, 2159], [50, 2159]])
TARGET_WIDTH = 12
TARGET_HEIGHT = 100
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

        if self.m is None:
            raise ValueError("Failed to initialize perspective transformation matrix.")
        else:
            print("Perspective transformation matrix:")
            print(self.m)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.m is None:
            raise ValueError("Perspective transformation matrix is not initialized.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)

        try:
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        except cv2.error as e:
            print(f"CV2 error: {e}")
            raise ValueError("Failed to transform points using perspective matrix.")

        if transformed_points is None:
            raise ValueError("Failed to transform points using perspective matrix.")
        else:
            print("Shapes - Input Points:", reshaped_points.shape, "Transformed Points:", transformed_points.shape)
            print("Input Points:")
            print(reshaped_points)
            print("Transformed Points:")
            print(transformed_points)

        return transformed_points.reshape(-1, 2)

# Defines a function that returns the source video path
def source_video_path():
    video_path = "Vehicle_Flow.mp4"
    confidence_threshold = 0.3
    iou_threshold = 0.7

    return video_path, confidence_threshold, iou_threshold

if __name__ == "__main__":
    video_path, confidence_threshold, iou_threshold = source_video_path()    # Gets video path and related threshold values
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)    # Retrieves video information
    model = YOLO("yolov8x.pt")    # YOLO loads our model according to the version we want (n, s, m, l, x).
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=confidence_threshold)    # Initializes the ByteTrack class
    thickness = sv.calculate_dynamic_line_thickness(resolution_wh=video_info.resolution_wh)    # Calculates line thickness
    text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)    # Calculates text scale
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)    # Draws a bounding box
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)    # Label inserter
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)    # Grid plotter
    frame_generator = sv.get_video_frames_generator(source_path=video_path)    # Gets a function that generates video frames
    polygon_zone = sv.PolygonZone(polygon=ROI_COORDINATES, frame_resolution_wh=video_info.resolution_wh)    # Identifies a specific region
    view_transformer = ViewTransformer(source=ROI_COORDINATES, target=TARGET)    # Starts the image converter
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))    # Tracks coordinates
    
    # Sets video recording parameters
    output_video_path = "Vehicle Speed Estimation.mp4"  # Name and extension of the video file to be recorded
    output_video_fps = video_info.fps  # Video fps value to record

    # Starts VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (Use 'mp4v' for .mp4 extension)
    out = cv2.VideoWriter(output_video_path, fourcc, output_video_fps, video_info.resolution_wh)

    # Processes each video frame
    for frame in frame_generator:
        # Detects objects using the YOLO model
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.confidence > confidence_threshold]  # Selects detections that exceed the credibility threshold
        detections = detections[polygon_zone.trigger(detections)]  # Selects detections within a given region
        detections = detections.with_nms(threshold=iou_threshold)  # Implements non-maximum suppression (NMS)
        detections = byte_track.update_with_detections(detections=detections)  # updates detections using ByteTrack

        # Transforms image coordinates with perspective transformation
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        # Saves coordinates of tracked objects
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # Calculates the speed of tracked objects and generates tags
        labels = []
        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"{int(speed)} km/h")

        # Labels the frame and adds the tracks
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("Vehicle Speed Estimation", annotated_frame)  # Shows the image
        out.write(annotated_frame)  # Adds the frame to VideoWriter

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Checks whether the 'q' key has been pressed to exit
            break

    out.release()  # Closes VideoWriter
    cv2.destroyAllWindows()  # Closes the OpenCV window
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

# Load MiDaS Model
def load_midas():
    model_type = "DPT_Large"  # Best quality model; use "MiDaS_small" for faster inference
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

# Run depth estimation
def estimate_depth(model, transform, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        depth = model(img)
    
    depth = depth.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (image.shape[1], image.shape[0]))  # Resize to original image size
    return depth

# Extract mean depth from bounding box
def get_bbox_depth(depth_map, bbox):
    x1, y1, x2, y2 = bbox
    roi = depth_map[y1:y2, x1:x2]
    mean_depth = np.mean(roi)  # Mean depth in bounding box
    return mean_depth

# Convert depth to real-world distance (if altitude is known)
def scale_depth(depth, drone_altitude, camera_angle):
    if drone_altitude is None:
        return depth  # No scaling if altitude is unknown

    # Estimate real-world distance using trigonometry
    real_distance = drone_altitude * np.tan(np.radians(camera_angle))
    scale_factor = real_distance / np.max(depth)  # Normalize depth map
    return depth * scale_factor

# Load and process an image
def process_image(image_path, bbox, drone_altitude=None, camera_angle=45):
    model, transform = load_midas()
    
    image = cv2.imread(image_path)
    depth_map = estimate_depth(model, transform, image)
    
    # Extract depth inside fire bounding box
    bbox_depth = get_bbox_depth(depth_map, bbox)
    
    # Scale if drone altitude is known
    real_distance = scale_depth(bbox_depth, drone_altitude, camera_angle)

    # Display results
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    text = f"Depth: {real_distance:.2f}m"
    cv2.putText(image, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Fire Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return real_distance

# Example usage
bbox = (100, 150, 300, 400)  # Example bounding box (x1, y1, x2, y2)
image_path = "fire_image.jpg"
drone_altitude = 50  # Example drone altitude in meters
camera_angle = 45  # Camera tilt angle in degrees

distance_to_fire = process_image(image_path, bbox, drone_altitude, camera_angle)
print(f"Estimated Distance to Fire: {distance_to_fire:.2f} meters")

import pyrealsense2 as rs
import numpy as np
import cv2
import json

# Configure depth and color streams at 1280x720
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics
color_profile = profile.get_stream(rs.stream.color)
depth_profile = profile.get_stream(rs.stream.depth)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

# Save intrinsics
intrinsics_data = {
    "color": {
        "width": color_intrinsics.width,
        "height": color_intrinsics.height,
        "fx": color_intrinsics.fx,
        "fy": color_intrinsics.fy,
        "ppx": color_intrinsics.ppx,
        "ppy": color_intrinsics.ppy,
        "model": str(color_intrinsics.model),
        "coeffs": color_intrinsics.coeffs
    },
    "depth": {
        "width": depth_intrinsics.width,
        "height": depth_intrinsics.height,
        "fx": depth_intrinsics.fx,
        "fy": depth_intrinsics.fy,
        "ppx": depth_intrinsics.ppx,
        "ppy": depth_intrinsics.ppy,
        "model": str(depth_intrinsics.model),
        "coeffs": depth_intrinsics.coeffs
    }
}

with open("camera_intrinsics.json", "w") as f:
    json.dump(intrinsics_data, f, indent=4)

print("Camera intrinsics saved.")

# Capture 5 frames
num_images = 5
for i in range(1, num_images + 1):
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame and color_frame:
            # Convert images
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit

            # Save color image
            cv2.imwrite(f"color_{i}.png", color_image)

            # Save depth image as 16-bit PNG (preserves depth in mm)
            cv2.imwrite(f"depth_{i}.png", depth_image)

            print(f"Saved color_{i}.png and depth_{i}.png")
            break

pipeline.stop()
print("Done capturing 5 frames.")
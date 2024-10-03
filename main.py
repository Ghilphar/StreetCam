import cv2
import datetime
import time

# Initialize the camera
camera = cv2.VideoCapture(0)  # 0 is the default USB camera

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set frame rate to 20 FPS
camera.set(cv2.CAP_PROP_FPS, 20)

# Define video codec (H264 for compression)
fourcc = cv2.VideoWriter_fourcc(*'H264')

# Define the region of interest (ROI) for motion detection
roi_x, roi_y, roi_width, roi_height = 250, 150, 200, 200

# Initialize background frame for the ROI
ret, first_frame = camera.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

# Crop the first frame to the region of interest (ROI)
first_gray_roi = first_gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

# Initialize clip numbering and motion tracking
clip_number = 1
motion_detected = False
motion_counter = 0  # Keeps track of how many frames motion is detected
no_motion_counter = 0  # Counts how many frames without motion

# Threshold for motion sensitivity
MOTION_THRESHOLD = 10  # Number of frames motion must be detected to start recording
NO_MOTION_THRESHOLD = 20  # Number of frames without motion to stop recording

# Function to create a new video writer object
def create_new_writer(clip_num):
    output_filename = f"clip_{clip_num:03d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    return cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480)), output_filename

# Initialize the big clip writer to record everything
big_clip_filename = f"full_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
big_clip_out = cv2.VideoWriter(big_clip_filename, fourcc, 20.0, (640, 480))

# Function to periodically update the background frame
def update_background(frame, interval=300):  # Update every 300 frames
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    return gray_frame

# Initialize first clip writer
out = None  # Start with no active motion clip
frame_count = 0  # Track the number of frames processed

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame_count += 1  # Increment frame count

    if ret:
        # Convert current frame to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Crop the current grayscale frame to the region of interest (ROI)
        gray_roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Compute the absolute difference between the current frame's ROI and the background frame's ROI
        frame_delta = cv2.absdiff(first_gray_roi, gray_roi)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes, then find contours
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for motion within the ROI (any detected contours indicate movement)
        if len(contours) > 0:
            # Motion detected, increment the motion counter
            motion_counter += 1
            no_motion_counter = 0  # Reset the no-motion counter

            # Start recording if motion persists
            if motion_counter > MOTION_THRESHOLD and not motion_detected:
                clip_number += 1
                out, filename = create_new_writer(clip_number)
                print(f"Motion detected, starting recording: {filename}")
                motion_detected = True

        else:
            # No motion detected, increment the no-motion counter
            no_motion_counter += 1

            # If no motion persists for a while, stop recording
            if no_motion_counter > NO_MOTION_THRESHOLD and motion_detected:
                print(f"No motion detected, stopping recording: {filename}")
                motion_detected = False
                if out is not None:
                    out.release()
                out = None
                motion_counter = 0  # Reset motion counter when stopping

        # Save frames to the big clip
        big_clip_out.write(frame)

        # Write to motion clip if motion is detected
        if motion_detected and out is not None:
            out.write(frame)

        # Draw the ROI as a red rectangle on the display frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)

        # Display the frame (optional)
        cv2.imshow('USB Camera', frame)

        # Update the background frame every 300 frames (controlled)
        if frame_count % 300 == 0:
            first_gray_roi = update_background(frame)[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

    else:
        print("Error: Failed to capture frame.")
        break

# Release everything once the job is finished
camera.release()
if out is not None:
    out.release()
big_clip_out.release()
cv2.destroyAllWindows()

import cv2
import datetime
import time
import argparse
import os
import smtplib
from email.mime.text import MIMEText

def initialize_camera(width=640, height=480, fps=20):
    """
    Initializes the camera with the given resolution and frame rate.
    """
    camera = cv2.VideoCapture(0)  # 0 is the default camera index
    if not camera.isOpened():
        raise IOError("Error: Could not open video device.")

    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)

    return camera

def create_video_writer(filename, fourcc, fps, frame_size, folder=''):
    """
    Creates and returns a VideoWriter object for recording video.
    Saves the file in the specified folder.
    """
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    return cv2.VideoWriter(filepath, fourcc, fps, frame_size)

def send_email_notification(subject, body, to_email):
    """
    Sends an email notification with the given subject and body to the specified email address.
    """
    # Email configuration (fill in your email details)
    smtp_server = 'smtp.example.com'
    smtp_port = 587
    from_email = 'your_email@example.com'
    password = 'your_password'

    # Create the email message
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Send the email via SMTP server
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        print(f"Email notification sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def parse_arguments():
    """
    Parses command-line arguments for motion sensitivity settings and ROI.
    """
    parser = argparse.ArgumentParser(description="Motion Detection Script")
    parser.add_argument('--motion_threshold', type=int, default=10, help='Frames of motion required to start recording.')
    parser.add_argument('--no_motion_threshold', type=int, default=20, help='Frames of no motion required to stop recording.')
    parser.add_argument('--sensitivity', type=int, default=500, help='Sensitivity for motion detection (lower is more sensitive).')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        default=[250, 150, 200, 200], help='Region of interest coordinates.')
    args = parser.parse_args()
    return args

def validate_roi(roi_coords, frame_width, frame_height):
    """
    Validates that the ROI coordinates are within the frame boundaries.
    """
    x, y, w, h = roi_coords
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return (x, y, w, h)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    motion_threshold = args.motion_threshold
    no_motion_threshold = args.no_motion_threshold
    sensitivity = args.sensitivity
    roi_coords = tuple(args.roi)

    # Parameters
    width, height = 640, 480
    fps = 20

    # Initialize camera and video writers
    camera = initialize_camera(width, height, fps)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    frame_size = (width, height)

    # Validate ROI coordinates
    roi_coords = validate_roi(roi_coords, width, height)
    x, y, w, h = roi_coords

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    # Initialize video writers
    clip_number = 1
    out = None  # Motion clip writer
    timestamp = datetime.datetime.now()
    date_folder = timestamp.strftime('%Y%m%d')
    big_clip_filename = f"full_{timestamp.strftime('%H%M%S')}.mp4"
    big_clip_out = create_video_writer(big_clip_filename, fourcc, fps, frame_size, folder=date_folder)

    # Motion detection variables
    motion_detected = False
    motion_counter = 0
    no_motion_counter = 0
    frame_count = 0
    email_sent = False
    to_email = 'recipient@example.com'  # Set the recipient email address

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            frame_count += 1

            # Crop to ROI
            roi_frame = frame[y:y + h, x:x + w]

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(roi_frame)

            # Apply threshold and morphological operations
            _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate the total area of motion
            motion_area = sum(cv2.contourArea(c) for c in contours)

            # Check for motion based on sensitivity
            if motion_area > sensitivity:
                motion_counter += 1
                no_motion_counter = 0
                if motion_counter > motion_threshold and not motion_detected:
                    clip_number += 1
                    timestamp = datetime.datetime.now()
                    date_folder = timestamp.strftime('%Y%m%d')
                    motion_filename = f"clip_{clip_number:03d}_{timestamp.strftime('%H%M%S')}.mp4"
                    out = create_video_writer(motion_filename, fourcc, fps, frame_size, folder=date_folder)
                    print(f"Motion detected, starting recording: {motion_filename}")
                    motion_detected = True
                    if not email_sent:
                        subject = "Motion Detected"
                        body = f"Motion detected at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
                        #send_email_notification(subject, body, to_email)
                        email_sent = True
            else:
                no_motion_counter += 1
                if no_motion_counter > no_motion_threshold and motion_detected:
                    print(f"No motion detected, stopping recording.")
                    motion_detected = False
                    if out:
                        out.release()
                        out = None
                    motion_counter = 0
                    email_sent = False  # Reset email notification

            # Write frames to video files
            big_clip_out.write(frame)
            if motion_detected and out:
                out.write(frame)

            # Draw ROI rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Motion Detection', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break

    except KeyboardInterrupt:
        print("Recording interrupted by user.")

    finally:
        # Release resources
        camera.release()
        if out:
            out.release()
        big_clip_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import pyttsx3
import datetime

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speech rate if needed

# Load the pre-trained data on car frontals from OpenCV
trained_car_data = cv2.CascadeClassifier('haarcascade_car.xml')

# Open the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

car_count = 0  # Initialize car count
log_file = open("car_detection_log.txt", "a")  # Log file to save detections

while True:
    # Read the current frame from the video stream
    _, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the grayscale frame
    car_coordinates = trained_car_data.detectMultiScale(gray_image)

    # Get frame dimensions and calculate grid size
    height, width, _ = frame.shape
    grid_width, grid_height = (width - 100) // 3, (height - 100) // 3

    # Draw a green square frame around the camera feed
    top_left = (50, 50)
    bottom_right = (width - 50, height - 50)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Draw grid lines and numbers for 3x3 division inside the green frame
    grid_number = 1
    for row in range(3):
        for col in range(3):
            cell_x = 50 + col * grid_width
            cell_y = 50 + row * grid_height
            cv2.rectangle(frame, (cell_x, cell_y), (cell_x + grid_width, cell_y + grid_height), (0, 255, 0), 1)
            text_x = cell_x + grid_width // 2 - 10
            text_y = cell_y + grid_height // 2 + 10
            cv2.putText(frame, str(grid_number), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            grid_number += 1

    # Track detected cars
    detected_grid_positions = set()  # Use a set to avoid duplicate positions
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Calculate the center point of the detected object
        car_center_x = x + w // 2
        car_center_y = y + h // 2

        # Determine the grid position (1 to 9)
        grid_col = (car_center_x - 50) // grid_width
        grid_row = (car_center_y - 50) // grid_height
        grid_position = int(grid_row * 3 + grid_col + 1)

        if 1 <= grid_position <= 9:
            detected_grid_positions.add(grid_position)
            car_count += 1  # Increment car count
            
            # Log detection
            log_file.write(f"{datetime.datetime.now()}: Car detected in grid {grid_position}\n")
            engine.say(f"Car detected in grid {grid_position}")
            engine.runAndWait()

    # Highlight grids where cars are detected
    for position in detected_grid_positions:
        row, col = divmod(position - 1, 3)
        cell_x = 50 + col * grid_width
        cell_y = 50 + row * grid_height
        cv2.rectangle(frame, (cell_x, cell_y), (cell_x + grid_width, cell_y + grid_height), (0, 255, 255), 2)  # Yellow highlight

    # Display car count on the frame
    cv2.putText(frame, f"Total Cars Detected: {car_count}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the processed frame
    cv2.imshow('Real-time Car Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
log_file.close()  # Close the log file

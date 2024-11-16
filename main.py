import cv2

# Load the pre-trained data on car frontals from OpenCV
trained_car_data = cv2.CascadeClassifier('haarcascade_car.xml')

# Open the default camera (usually the webcam)
video_capture = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video stream
    _, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the grayscale frame
    car_coordinates = trained_car_data.detectMultiScale(gray_image)

    # Draw rectangles around the detected cars and display their names
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw a green square frame around the camera feed
    height, width, _ = frame.shape
    top_left = (50, 50)
    bottom_right = (width - 50, height - 50)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Display the frame with car detections and the green square frame
    cv2.imshow('Real-time Car Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()

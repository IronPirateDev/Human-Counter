import cv2
import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="dpsbn",
    database="test"
)
cursor = db.cursor()

# Initialize counters
enter_count = 0
exit_count = 0
total_count = 0

# Initialize background subtraction model
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Open camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Ignore small contours
        if area < 1000:
            continue
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate centroid
        centroid_x = int(x + w / 2)
        centroid_y = int(y + h / 2)
        
        # Draw bounding box and centroid
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (centroid_x, centroid_y), 2, (0, 0, 255), -1)
        
        # Update counters based on centroid position
        if centroid_y > 300:
            enter_count += 1
            total_count += 1
        elif centroid_y < 200:
            exit_count += 1
            total_count -= 1
    
    # Update the counts in the MySQL database
    query = "UPDATE people_counts SET enter_count = %s, exit_count = %s, total_count = %s WHERE id = 1"
    values = (enter_count, exit_count, total_count)
    cursor.execute(query, values)
    db.commit()
    
    # Display the frame with bounding boxes and counts
    cv2.putText(frame, f"Enter: {enter_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Total: {total_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('People Count', frame)
    
    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
db.close()

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize Mediapipe Hands and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# OpenCV Setup
cap = cv2.VideoCapture(0)
canvas = None  # Canvas for drawing
undo_stack = []  # Stack for undo actions
redo_stack = []  # Stack for redo actions
is_drawing = False
is_erasing = False
brush_color = (255, 0, 0)
brush_size = 4
eraser_size = 50
last_x, last_y = None, None


def generate_frames():
    """Generate frames for video streaming with drawing and erasing functionality."""
    global canvas, is_drawing, is_erasing, last_x, last_y, brush_color, brush_size, eraser_size

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Initialize the canvas if not already created
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert the frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index finger tip (Landmark 8)
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

                # Detect the number of visible fingers
                visible_fingers = sum(
                    hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y
                    for i in [4, 8, 12, 16, 20]
                )

                # Toggle modes based on visible fingers
                if visible_fingers >= 4:  # More than 4 fingers visible => Erasing mode
                    is_erasing = True
                    is_drawing = False
                else:  # Less than 4 fingers visible => Drawing mode
                    is_erasing = False
                    is_drawing = True

                # Drawing and erasing functionality
                if is_erasing:
                    if canvas is not None:
                        undo_stack.append(canvas.copy())  # Save canvas state for undo
                        cv2.circle(canvas, (x, y), eraser_size, (0, 0, 0), -1)
                elif is_drawing and last_x is not None and last_y is not None:
                    if canvas is not None:
                        undo_stack.append(canvas.copy())  # Save canvas state for undo
                        cv2.line(canvas, (last_x, last_y), (x, y), brush_color, thickness=brush_size)

                # Update last positions
                last_x, last_y = x, y
        else:
            is_drawing = False
            is_erasing = False

        # Blend the video feed and canvas
        combined = cv2.addWeighted(frame, 0.9, canvas, 1, 0)

        # Display mode text on the frame
        mode_text = f"Mode: {'Erasing' if is_erasing else 'Drawing' if is_drawing else 'Idle'}"
        cv2.putText(combined, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', combined)
        combined_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')


@app.route("/")
def index():
    """Render the main HTML interface."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/update_settings", methods=["POST"])
def update_settings():
    """Update brush size and color."""
    global brush_size, brush_color
    data = request.json
    if "brush_size" in data:
        brush_size = int(data["brush_size"])
    if "brush_color" in data:
        brush_color = tuple(int(data["brush_color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return jsonify({"status": "success"})


@app.route("/reset_canvas", methods=["POST"])
def reset_canvas():
    """Reset the canvas."""
    global canvas
    canvas = None
    return jsonify({"status": "reset"})


@app.route("/undo", methods=["POST"])
def undo():
    """Undo the last drawing action."""
    global canvas, undo_stack, redo_stack
    if undo_stack:
        redo_stack.append(canvas.copy())  # Save the current state for redo
        canvas = undo_stack.pop()
        return jsonify({"status": "undone"})
    return jsonify({"status": "error", "message": "Nothing to undo"})


@app.route("/redo", methods=["POST"])
def redo():
    """Redo the last undone action."""
    global canvas, undo_stack, redo_stack
    if redo_stack:
        undo_stack.append(canvas.copy())  # Save the current state for undo
        canvas = redo_stack.pop()
        return jsonify({"status": "redone"})
    return jsonify({"status": "error", "message": "Nothing to redo"})


@app.route("/save_canvas", methods=["GET"])
def save_canvas():
    """Save the canvas as an image."""
    global canvas
    if canvas is not None:
        filename = "static/saved_canvas.png"
        cv2.imwrite(filename, canvas)  # Save the canvas as a PNG file
        return jsonify({"status": "saved", "filename": filename})
    return jsonify({"status": "error", "message": "Canvas is empty"})


if __name__ == "__main__":
    app.run(debug=True)

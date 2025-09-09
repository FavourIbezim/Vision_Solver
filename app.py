from flask import Flask, render_template, Response, jsonify
import cv2
import base64
import time
import google.generativeai as genai

app = Flask(__name__)

# 🔑 Your Gemini API key
API_KEY = "AIzaSyAU2dQV_57OgkaCVFSvdXMBLMjyCdeZyAo"
genai.configure(api_key=API_KEY)

# Load model
model = genai.GenerativeModel("gemini-1.5-flash")

cap = cv2.VideoCapture(0)
last_analysis = "Analyzing..."
last_time = 0
interval = 3  # seconds between API calls


def generate_frames():
    global last_analysis, last_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize for faster inference
        small_frame = cv2.resize(frame, (640, 480))

        # Run analysis every few seconds
        if time.time() - last_time > interval:
            _, buffer = cv2.imencode(".jpg", small_frame)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            try:
                response = model.generate_content([
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}},
                    {"text": "What problem does this image represent? Give a solution."}
                ])
                last_analysis = response.text
            except Exception as e:
                last_analysis = f"Error: {e}"

            last_time = time.time()

        # Encode frame for streaming
        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/analysis')
def analysis():
    return jsonify({"analysis": last_analysis})


if __name__ == '__main__':
    app.run(debug=True)

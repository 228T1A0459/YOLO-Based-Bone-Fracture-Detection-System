from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "bone_fracture_secret"

UPLOAD_FOLDER = r"C:\project\Bone fature detection-final\Bone fature detection\uploads"
MODEL_PATH = r"C:\project\Bone fature detection-final\Bone fature detection\best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = YOLO(MODEL_PATH)
model = YOLO(MODEL_PATH)

CLASS_LABELS = {
    "hi": "Bone Fracture Detected",
    "hello": "Normal Bone"
}

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "password":
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# ---------------- HOME (Protected) ----------------
@app.route("/home", methods=["GET", "POST"])
def home():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- UPLOADS ----------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------------- PREDICTION ----------------
@app.route("/upload", methods=["POST"])
def upload():
    if not session.get("logged_in"):
        return jsonify({"error": "Unauthorized"}), 401

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    results = model(file_path)

    detected = False
    details = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) * 100
            cls = r.names[int(box.cls[0])]

            # Convert model class name to user-friendly name
            display_cls = CLASS_LABELS.get(cls, cls)

            detected = True
            details.append(f"{display_cls} ({conf:.2f}%)")

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"{display_cls} {conf:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    output_img = "detected_" + file.filename
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, output_img), image)

    return jsonify({
        "image": output_img,
        "status": "Fracture Detected" if detected else "No Fracture",
        "details": ", ".join(details) if details else "Healthy Bone"
    })


if __name__ == "__main__":
     app.run(debug=True)

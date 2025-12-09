import os
import json
import numpy as np
import cv2
import mediapipe as mp
import psycopg2
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEDIA PIPE SETUP (For Head Pose) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- DATABASE ---
def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

@app.on_event("startup")
def startup_event():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                full_name TEXT NOT NULL,
                embedding FLOAT[] 
            );
        """)
        conn.commit()
        conn.close()
        print("✅ Database ready.")
    except Exception as e:
        print(f"❌ DB Error: {e}")

# --- HELPER: GET EMBEDDING ---
def get_embedding(img_path):
    try:
        results = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="opencv",
            align=True
        )
        return results[0]["embedding"]
    except:
        return None

# --- HELPER: ESTIMATE POSE (FIXED DIRECTIONS) ---
def estimate_pose(img_path):
    image = cv2.imread(img_path)
    if image is None: return "No Face"
    
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return "No Face"

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    
    # Landmarks: 1=Nose, 33=Left Eye, 263=Right Eye
    nose_tip = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    # Calculate X coordinates
    nx = nose_tip.x * w
    lx = left_eye.x * w
    rx = right_eye.x * w

    dist_to_left_eye = abs(lx - nx)
    dist_to_right_eye = abs(rx - nx)
    
    total_dist = dist_to_left_eye + dist_to_right_eye
    if total_dist == 0: return "Unknown"
    
    ratio = dist_to_left_eye / total_dist
    
    # --- SWAPPED LOGIC FOR MIRRORED SELFIE ---
    if 0.35 <= ratio <= 0.65:
        return "Front"
    elif ratio < 0.35:
        return "Right"
    elif ratio > 0.65:
        return "Left"
    
    return "Unknown"

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Backend Running"}

# NEW: Check Pose Endpoint
@app.post("/check-pose")
async def check_pose(target: str = Form(...), file: UploadFile = File(...)):
    # Save temp
    temp = f"temp_pose_{file.filename}"
    with open(temp, "wb") as f:
        f.write(await file.read())
    
    detected_pose = estimate_pose(temp)
    
    if os.path.exists(temp):
        os.remove(temp)
        
    is_correct = (detected_pose == target)
    
    # Allow 'Front' to match loosely if we just want a face
    if target == "Front" and detected_pose in ["Front", "Left", "Right"]:
        pass 

    return {
        "correct": is_correct, 
        "detected": detected_pose, 
        "message": f"Looking {detected_pose}"
    }

# UPDATED: Register with 3 Images
@app.post("/register")
async def register(name: str = Form(...), files: List[UploadFile] = File(...)):
    embeddings = []
    
    for file in files:
        temp = f"temp_reg_{file.filename}"
        with open(temp, "wb") as f:
            f.write(await file.read())
        
        emb = get_embedding(temp)
        if os.path.exists(temp):
            os.remove(temp)
            
        if emb:
            embeddings.append(emb)

    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid faces found in any photo")

    # Average the embeddings for better accuracy
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (full_name, embedding) VALUES (%s, %s)",
        (name, avg_embedding)
    )
    conn.commit()
    conn.close()
    
    return {"message": f"User {name} registered with {len(embeddings)} angles!"}

# --- MODIFIED VERIFY ENDPOINT ---
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp = f"verify_{file.filename}"
    with open(temp, "wb") as f:
        f.write(await file.read())

    target_emb = get_embedding(temp)
    if os.path.exists(temp):
        os.remove(temp)

    # --- CHANGE START ---
    # Previously, this raised an HTTPException(400) which showed the error popup.
    # Now, we simply return a clean JSON saying match=False.
    if not target_emb:
        # We return a successful HTTP status (200), but logic indicates no face found.
        # This keeps the frontend silent.
        return {"match": False, "user": "No Face", "distance": 1.0}
    # --- CHANGE END ---

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT full_name, embedding FROM users")
    rows = cur.fetchall()
    conn.close()

    best_match = "Unknown"
    min_dist = 100.0
    target_np = np.array(target_emb)

    for name, db_emb_list in rows:
        db_np = np.array(db_emb_list)
        similarity = np.dot(target_np, db_np) / (np.linalg.norm(target_np) * np.linalg.norm(db_np))
        dist = 1 - similarity
        if dist < min_dist:
            min_dist = dist
            best_match = name

    if min_dist < 0.30:
        return {"match": True, "user": best_match, "distance": float(min_dist)}
    
    return {"match": False, "user": "Unknown", "distance": float(min_dist)}
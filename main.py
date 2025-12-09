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

# --- HELPER: ESTIMATE POSE ---
def estimate_pose(img_path):
    """
    Returns 'Front', 'Left', 'Right', or 'Unknown'
    """
    image = cv2.imread(img_path)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return "No Face"

    # Get key landmarks
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    
    # 1 (Nose Tip), 33 (Left Eye Outer), 263 (Right Eye Outer)
    nose_tip = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    # Convert to pixels
    nx, ny = nose_tip.x * w, nose_tip.y * h
    lx, ly = left_eye.x * w, left_eye.y * h
    rx, ry = right_eye.x * w, right_eye.y * h

    # Calculate distances
    dist_left = nx - lx  # Nose to Left Eye
    dist_right = rx - nx # Nose to Right Eye
    
    # Ratio approach
    total_dist = dist_left + dist_right
    if total_dist == 0: return "Unknown"
    
    ratio = dist_left / total_dist
    
    # Thresholds (Adjust these if needed)
    # 0.5 is perfectly centered. 
    # < 0.35 means nose is very close to left eye (Looking Left)
    # > 0.65 means nose is very close to right eye (Looking Right)
    
    if 0.40 <= ratio <= 0.60:
        return "Front"
    elif ratio < 0.40:
        return "Left" # User is looking to their right (Camera Left)
    elif ratio > 0.60:
        return "Right" # User is looking to their left (Camera Right)
    
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
        # Strict mode: Only true Front
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

# VERIFY (Unchanged)
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp = f"verify_{file.filename}"
    with open(temp, "wb") as f:
        f.write(await file.read())

    target_emb = get_embedding(temp)
    if os.path.exists(temp):
        os.remove(temp)

    if not target_emb:
        raise HTTPException(status_code=400, detail="No face detected")

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
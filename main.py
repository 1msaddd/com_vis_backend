import os
import json
import numpy as np
import psycopg2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

app = FastAPI()

# Allow Next.js to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONNECTION ---
def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# --- INIT DB (Run once on startup) ---
@app.on_event("startup")
def startup_event():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Standard table. "embedding" is just an array of float numbers.
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

# --- ROUTES ---
@app.get("/")
def home():
    return {"status": "Backend Running"}

@app.post("/register")
async def register(name: str, file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    embedding = get_embedding(temp_filename)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if not embedding:
        raise HTTPException(status_code=400, detail="No face detected")

    conn = get_db_connection()
    cur = conn.cursor()
    # Save array as a list
    cur.execute(
        "INSERT INTO users (full_name, embedding) VALUES (%s, %s)",
        (name, list(embedding))
    )
    conn.commit()
    conn.close()
    
    return {"message": f"User {name} registered!"}

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    temp_filename = f"verify_{file.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    target_emb = get_embedding(temp_filename)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    if not target_emb:
        raise HTTPException(status_code=400, detail="No face detected")

    # 1. Fetch ALL users
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT full_name, embedding FROM users")
    rows = cur.fetchall()
    conn.close()

    # 2. Compare in Python
    best_match = "Unknown"
    min_dist = 100.0
    target_np = np.array(target_emb)

    for name, db_emb_list in rows:
        db_np = np.array(db_emb_list)
        
        # Cosine Distance Formula
        similarity = np.dot(target_np, db_np) / (np.linalg.norm(target_np) * np.linalg.norm(db_np))
        dist = 1 - similarity
        
        if dist < min_dist:
            min_dist = dist
            best_match = name

    # 3. Threshold check (0.30 for FaceNet512)
    if min_dist < 0.30:
        return {"match": True, "user": best_match, "distance": float(min_dist)}
    
    return {"match": False, "user": "Unknown", "distance": float(min_dist)}
import sqlite3
import os
import pandas as pd
from datetime import date, datetime

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "attendance.db")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "attendance"), exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT UNIQUE NOT NULL,
            photo_path TEXT,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            name TEXT,
            roll_no TEXT,
            date TEXT,
            time TEXT,
            confidence REAL,
            status TEXT DEFAULT 'Present',
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized")


def get_all_students():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM students ORDER BY name", conn)
    conn.close()
    return df


def get_attendance(date_filter=None):
    conn = get_connection()
    if date_filter:
        df = pd.read_sql_query(
            "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC",
            conn, params=(date_filter,)
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM attendance ORDER BY date DESC, time DESC LIMIT 200",
            conn
        )
    conn.close()
    return df


def mark_attendance(name, roll_no, confidence):
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    c = conn.cursor()

    # Already marked today?
    c.execute(
        "SELECT id FROM attendance WHERE roll_no = ? AND date = ?",
        (roll_no, today)
    )
    if c.fetchone():
        conn.close()
        return False, "Already marked today"

    c.execute(
        "INSERT INTO attendance (name, roll_no, date, time, confidence, status) VALUES (?, ?, ?, ?, ?, ?)",
        (name, roll_no, today, now, confidence, "Present")
    )
    conn.commit()
    conn.close()
    return True, f"Attendance marked for {name} at {now}"


def register_student(name, roll_no, photo_path):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO students (name, roll_no, photo_path) VALUES (?, ?, ?)",
            (name, roll_no, photo_path)
        )
        conn.commit()
        conn.close()
        return True, "Student registered successfully!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Roll number already exists"


# ================== EXCEL MANAGEMENT ==================

def export_students_to_excel():
    """Students table ko Excel bytes mein return karta hai"""
    df = get_all_students()
    if df.empty:
        return None
    # Sirf useful columns
    export_df = df[["name", "roll_no", "registered_at"]].copy()
    export_df.columns = ["Name", "Roll No", "Registered At"]
    
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Students")
    return output.getvalue()


def export_attendance_to_excel(date_filter=None):
    """Attendance records ko Excel bytes mein return karta hai"""
    df = get_attendance(date_filter)
    if df.empty:
        return None
    export_df = df[["name", "roll_no", "date", "time", "confidence", "status"]].copy()
    export_df.columns = ["Name", "Roll No", "Date", "Time", "Confidence %", "Status"]
    
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Attendance")
    return output.getvalue()


def import_students_from_excel(uploaded_file):
    """
    Excel se bulk students add karta hai.
    Expected columns: Name, Roll No
    Returns (success_count, error_messages)
    """
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        return 0, [f"Excel read error: {str(e)}"]
    
    # Column names flexible banao
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    if "name" not in df.columns or "roll_no" not in df.columns:
        return 0, ["Excel mein 'Name' aur 'Roll No' columns hone chahiye"]
    
    success = 0
    errors = []
    
    for idx, row in df.iterrows():
        name = str(row["name"]).strip()
        roll = str(row["roll_no"]).strip()
        
        if not name or not roll or name == "nan" or roll == "nan":
            errors.append(f"Row {idx+2}: Name ya Roll No empty")
            continue
        
        # Photo path empty rakhte hain (baad mein photo add kar sakte ho)
        ok, msg = register_student(name, roll, photo_path="")
        if ok:
            success += 1
        else:
            errors.append(f"{name} ({roll}): {msg}")
    
    return success, errors


def delete_student(roll_no):
    """
    Student delete karta hai + uski saari attendance bhi delete.
    Returns (success: bool, message: str)
    """
    conn = get_connection()
    c = conn.cursor()

    # Check if student exists
    c.execute("SELECT name FROM students WHERE roll_no = ?", (roll_no,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "Student not found"

    name = row[0]

    # Delete attendance first
    c.execute("DELETE FROM attendance WHERE roll_no = ?", (roll_no,))
    att_deleted = c.rowcount

    # Delete student
    c.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
    conn.commit()
    conn.close()

    # Try to delete photo file if exists
    try:
        students = get_all_students()  # refresh not needed, just for path logic
        # photo path was stored, but we already deleted row — try common pattern
        import glob
        for f in glob.glob(os.path.join(KNOWN_FACES_DIR, f"*_{roll_no}.jpg")):
            os.remove(f)
        for f in glob.glob(os.path.join(KNOWN_FACES_DIR, f"*_{roll_no}.png")):
            os.remove(f)
    except Exception:
        pass

    return True, f"Deleted {name} ({roll_no}) and {att_deleted} attendance record(s)"

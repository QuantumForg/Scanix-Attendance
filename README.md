# AI-Based Smart Attendance Monitoring System

**IBM Project** | Frontend + Backend Separated Architecture

Built with **Python + Streamlit (Frontend) + DeepFace + Groq**

---

## Architecture

```
AI_Smart_Attendance_System/
├── app.py                     # FRONTEND (Streamlit UI only)
├── backend/
│   ├── database.py            # SQLite operations
│   ├── face_engine.py         # DeepFace recognition + photo saving
│   └── ai_insights.py         # Groq LLM insights
├── data/
│   ├── known_faces/           # Student photos
│   └── attendance.db          # Auto-created
├── requirements.txt
└── README.md
```

**Frontend** handles only UI, buttons, charts, camera.  
**Backend** modules contain all logic (DB, AI, face matching).

---

## Features

- Real face recognition (DeepFace VGG-Face)
- Student registration + attendance marking
- SQLite database
- Beautiful dark modern UI
- Dashboard, Records, CSV export
- Optional Groq AI insights
- Graceful fallbacks (no crash without API key)

---

## Quick Start

```bash
cd AI_Smart_Attendance_System
pip install -r requirements.txt
streamlit run app.py
```

Optional:
```bash
export GROQ_API_KEY="your_key"
```

---

## Backend Modules (Viva Explanation)

| Module | Responsibility |
|--------|----------------|
| `database.py` | init_db, register_student, mark_attendance, get records |
| `face_engine.py` | save photo, recognize_face using DeepFace |
| `ai_insights.py` | Call Groq LLM for natural language report |

Frontend simply imports these functions and displays results.

---

**IBM Project List – Project #1**

## Excel Management Feature

Naya page: **📁 Excel Management**

- Students list download as Excel
- Attendance records download as Excel  
- Bulk students import from Excel (Name + Roll No)
- Sample template download

Isse database manage karna Sir ke liye bahut easy ho jaata hai.

## New Features Added

### 📅 Timetable & Auto Attendance
- Daily 8 lectures schedule (50 min each)
- Auto trigger time = class end - 5 minutes
- Simulation button for demo

### 📹 Basic RTSP / CCTV Support
- Mark Attendance page mein RTSP URL option
- OpenCV se frames capture
- Classroom IP cameras ke liye ready

### 🚀 Future Scope Page
- College ERP integration vision
- Automatic CCTV attendance architecture
- Scheduler + Excel + ERP update flow

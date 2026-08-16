import streamlit as st
import numpy as np
import pandas as pd
from datetime import date, datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root so backend imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.database import init_db, get_all_students, get_attendance, mark_attendance, register_student, delete_student
from backend.face_engine import save_student_photo, recognize_face, DEEPFACE_OK
from backend.ai_insights import get_ai_insight, GROQ_OK

# ================== PAGE CONFIG ==================
# Must be first Streamlit command
st.set_page_config(
    page_title="Scanix | AI Smart Attendance",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Init DB after page config
try:
    init_db()
except Exception as e:
    st.error(f"Database init failed: {e}")


# ================== DESIGN SYSTEM ==================
# Identity: "Biometric Scan" — a face-recognition system should *feel* like
# one: a scanning laser sweep as the signature motif, viewfinder corner
# brackets around every camera/photo surface, and a mono data face for
# every number that comes off a sensor (confidence %, timestamps, roll no).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-0: #05070d;
    --bg-1: #0a0f1c;
    --surface: #101626;
    --border: rgba(148, 163, 184, 0.14);
    --border-hi: rgba(0, 245, 160, 0.45);
    --accent: #00f5a0;
    --cyan: #38bdf8;
    --text: #eef2fa;
    --muted: #8a93ab;
    --danger: #ff5c7a;
    --warn: #ffb454;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Solid dark background — no fixed overlays (those caused black screen on Windows) */
.stApp {
    background: linear-gradient(180deg, #05070d 0%, #0a0f1c 100%);
    color: #eef2fa;
}

/* Sidebar — solid, no backdrop-filter */
section[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid var(--border);
    min-width: 260px;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #eef2fa !important;
}

/* Brand */
.brand-mark { display: flex; align-items: center; gap: 10px; padding: 4px 0; }
.brand-ring {
    width: 34px; height: 34px; border-radius: 50%;
    border: 2px solid var(--accent);
    box-shadow: 0 0 14px rgba(0,245,160,0.45);
    position: relative; flex-shrink: 0;
}
.brand-ring::after {
    content: ''; position: absolute; inset: 8px;
    border-radius: 50%; background: var(--accent);
}
.brand-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.15rem; color: #eef2fa; }
.brand-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--accent); letter-spacing: 1.5px; text-transform: uppercase; }

/* Status pills */
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 5px 10px; border-radius: 20px; width: 100%;
    margin-bottom: 6px;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-on { background: rgba(0,245,160,0.1); border: 1px solid rgba(0,245,160,0.35); color: var(--accent); }
.status-on .status-dot { background: var(--accent); }
.status-off { background: rgba(255,92,122,0.1); border: 1px solid rgba(255,92,122,0.35); color: var(--danger); }
.status-off .status-dot { background: var(--danger); }
.status-mid { background: rgba(255,180,84,0.1); border: 1px solid rgba(255,180,84,0.35); color: var(--warn); }
.status-mid .status-dot { background: var(--warn); }

/* Headers */
.hero-wrap { margin-bottom: 1.5rem; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: var(--accent); letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 6px;
}
.main-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #eef2fa !important; margin: 0 0 4px 0;
}
.sub-header { color: #8a93ab !important; font-size: 1rem; margin: 0; }

/* Metric cards */
.metric-card {
    background: #101626;
    border-radius: 16px;
    padding: 20px 22px;
    border: 1px solid var(--border);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-hi);
}
.metric-icon { font-size: 1.1rem; margin-bottom: 8px; }
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 600; margin: 0;
    color: #00f5a0 !important;
}
.metric-label { font-size: 0.82rem; color: #8a93ab; margin-top: 5px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00c98a) !important;
    color: #04140d !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 16px rgba(0,245,160,0.25);
}
.stButton > button:hover {
    filter: brightness(1.08);
    box-shadow: 0 8px 22px rgba(0,245,160,0.35);
}
.stButton > button p { color: #04140d !important; font-weight: 700 !important; }

.stDownloadButton > button {
    background: rgba(0,245,160,0.1) !important;
    color: #00f5a0 !important;
    border: 1px solid rgba(0,245,160,0.4) !important;
    border-radius: 10px !important;
}

/* Boxes */
.success-box {
    background: rgba(0,245,160,0.1);
    border: 1px solid rgba(0,245,160,0.3);
    padding: 13px 16px; border-radius: 12px; margin: 8px 0;
    color: #b7ffe4;
}
.info-box {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    padding: 13px 16px; border-radius: 12px; margin: 8px 0;
    color: #cdefff;
}

.section-divider {
    height: 1px; margin: 1.2rem 0;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}
.foot-note {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: #8a93ab; text-align: center; letter-spacing: 1px;
}

#MainMenu, footer, header { visibility: hidden; }
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }

/* Ensure main content is visible */
.block-container, [data-testid="stAppViewContainer"], .main {
    color: #eef2fa;
}

/* ===== Biometric scan animation (safe, local only) ===== */
.scan-frame {
    position: relative;
    display: inline-block;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,245,160,0.35);
    box-shadow: 0 0 0 1px rgba(0,245,160,0.1), 0 8px 24px rgba(0,0,0,0.35);
}
.scan-frame img {
    display: block;
    border-radius: 12px;
}
/* laser line */
.scan-frame::after {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00f5a0, #fff, #00f5a0, transparent);
    box-shadow: 0 0 16px 4px rgba(0,245,160,0.55);
    animation: scanSweep 2.2s ease-in-out infinite;
    pointer-events: none;
    z-index: 2;
}
@keyframes scanSweep {
    0%   { top: 0%; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
/* viewfinder corners */
.scan-frame::before {
    content: '';
    position: absolute;
    inset: 8px;
    border: 2px solid transparent;
    border-image: none;
    pointer-events: none;
    z-index: 3;
    box-shadow:
        -12px -12px 0 -10px #00f5a0,
         12px -12px 0 -10px #00f5a0,
        -12px  12px 0 -10px #00f5a0,
         12px  12px 0 -10px #00f5a0;
}
.scan-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00f5a0;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 8px;
    text-align: center;
}
.scan-chip {
    display: inline-block;
    width: 28px; height: 2px;
    background: #00f5a0;
    margin-right: 8px;
    vertical-align: middle;
    position: relative;
    overflow: hidden;
}
.scan-chip::after {
    content: '';
    position: absolute;
    top: 0; left: -40%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, #fff, transparent);
    animation: chipSweep 1.6s ease-in-out infinite;
}
@keyframes chipSweep {
    0% { left: -40%; }
    100% { left: 100%; }
}

</style>
""", unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str):
    """Consistent page header: mono eyebrow with the scan-sweep signature, then title + subtitle."""
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow"><span class="scan-chip"></span>{eyebrow}</div>
        <p class="main-header">{title}</p>
        <p class="sub-header">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon: str, value, label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def themed_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#8a93ab",
        font_family="Inter",
        margin=dict(t=24, b=20, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)")
    return fig


# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("""
    <div class="brand-mark">
        <div class="brand-ring"></div>
        <div>
            <div class="brand-title">Scanix</div>
            <div class="brand-sub">Smart Attendance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📝 Register Student", "✅ Mark Attendance", "📊 Records & Reports", "📁 Excel Management", "📅 Timetable & Auto", "🚀 Future Scope", "🤖 AI Insights"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#8a93ab;letter-spacing:2px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)

    if DEEPFACE_OK:
        st.markdown("<div class='status-pill status-on'><span class='status-dot'></span>Face Engine Online</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-pill status-off'><span class='status-dot'></span>Face Engine Missing</div>", unsafe_allow_html=True)

    if GROQ_OK and os.environ.get("GROQ_API_KEY"):
        st.markdown("<div class='status-pill status-on'><span class='status-dot'></span>Groq AI Online</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-pill status-mid'><span class='status-dot'></span>Groq AI Optional</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.caption("Frontend: Streamlit · Backend: database + face_engine + ai_insights")

# ================== PAGES ==================

if page == "🏠 Dashboard":
    hero("LIVE OVERVIEW", "AI-Based Smart Attendance", "Real-time face recognition attendance monitoring system")

    students = get_all_students()
    attendance = get_attendance()
    today_str = date.today().isoformat()
    today_att = attendance[attendance["date"] == today_str] if not attendance.empty else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("👥", len(students), "Registered Students")
    with col2:
        metric_card("✅", len(today_att), "Present Today")
    with col3:
        absent = max(0, len(students) - len(today_att))
        metric_card("⭘", absent, "Absent Today")
    with col4:
        rate = (len(today_att) / len(students) * 100) if len(students) > 0 else 0
        metric_card("◎", f"{rate:.0f}%", "Attendance Rate")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Recent Attendance Trend")
        if not attendance.empty:
            daily = attendance.groupby("date").size().reset_index(name="count").tail(14)
            fig = px.bar(daily, x="date", y="count", color_discrete_sequence=["#00f5a0"])
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(themed_chart(fig), use_container_width=True)
        else:
            st.info("No attendance data yet.")
    with c2:
        st.subheader("✅ Today's Present Students")
        if not today_att.empty:
            st.dataframe(
                today_att[["name", "roll_no", "time", "confidence"]].rename(
                    columns={"name": "Name", "roll_no": "Roll No", "time": "Time", "confidence": "Confidence %"}
                ),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No one marked present yet today.")

elif page == "📝 Register Student":
    hero("ENROLLMENT", "Register New Student", "Add student photo for face recognition enrollment")

    col1, col2 = st.columns([1, 1])
    with col1:
        name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma")
        roll_no = st.text_input("Roll Number", placeholder="e.g. CS2025001")
        st.markdown("#### Capture or Upload Photo")
        source = st.radio("Photo Source", ["📷 Camera", "📁 Upload Image"], horizontal=True)

        image = None
        if source == "📷 Camera":
            cam = st.camera_input("Take a clear frontal photo")
            if cam:
                image = Image.open(cam)
        else:
            uploaded = st.file_uploader("Upload clear face photo", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)

        if image is not None:
            import base64
            from io import BytesIO
            buf = BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="scan-frame">
                <img src="data:image/jpeg;base64,{b64}" width="300" style="max-width:100%;" />
            </div>
            <div class="scan-label">◉ ENROLLMENT PREVIEW</div>
            """, unsafe_allow_html=True)

        if st.button("🚀 Register Student", type="primary", use_container_width=True):
            if not name or not roll_no:
                st.error("Please enter name and roll number")
            elif image is None:
                st.error("Please capture or upload a photo")
            else:
                with st.spinner("Registering..."):
                    photo_path = save_student_photo(name.strip(), roll_no.strip(), image)
                    ok, msg = register_student(name.strip(), roll_no.strip(), photo_path)
                    if ok:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)

    with col2:
        st.subheader("Already Registered")
        students = get_all_students()
        if not students.empty:
            for _, row in students.iterrows():
                c1, c2, c3 = st.columns([1, 2.2, 1])
                with c1:
                    if row["photo_path"] and os.path.exists(str(row["photo_path"])):
                        st.image(row["photo_path"], width=70)
                    else:
                        st.markdown("👤")
                with c2:
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"Roll: {row['roll_no']}")
                with c3:
                    if st.button("🗑️", key=f"del_{row['roll_no']}", help=f"Delete {row['name']}"):
                        ok, msg = delete_student(str(row['roll_no']))
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        else:
            st.info("No students registered yet.")

elif page == "✅ Mark Attendance":
    hero("FACE SCAN", "Mark Attendance", "Capture face → AI recognizes → Attendance marked automatically")

    if not DEEPFACE_OK:
        st.error("DeepFace library is required. Please install: pip install deepface")
        st.stop()

    students = get_all_students()
    if students.empty:
        st.warning("No students registered. Go to Register Student first.")
        st.stop()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Capture Face for Attendance")
        source = st.radio("Source", ["📷 Live Camera", "📁 Upload Photo", "📹 RTSP / CCTV Camera"], horizontal=True, key="att_source")

        query_img = None
        rtsp_frames = []

        if source == "📷 Live Camera":
            cam = st.camera_input("Look at the camera clearly", key="att_cam")
            if cam:
                query_img = Image.open(cam)
        elif source == "📁 Upload Photo":
            up = st.file_uploader("Upload a photo with face", type=["jpg", "jpeg", "png"], key="att_up")
            if up:
                query_img = Image.open(up)
        else:
            st.markdown("#### RTSP Camera (CCTV / IP Camera)")
            st.caption("Example: rtsp://username:password@192.168.1.64:554/stream1")
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:port/stream")
            if rtsp_url and st.button("Connect & Capture Frames", key="rtsp_btn"):
                with st.spinner("Connecting to camera... (timeout 8 sec)"):
                    from backend.face_engine import capture_from_rtsp
                    rtsp_frames = capture_from_rtsp(rtsp_url.strip(), max_frames=4)
                    if rtsp_frames:
                        st.success(f"Captured {len(rtsp_frames)} frames")
                        query_img = Image.fromarray(rtsp_frames[0])
                        st.session_state["rtsp_frames"] = rtsp_frames
                    else:
                        st.error("Could not connect to RTSP camera. Check URL / network.")
            if "rtsp_frames" in st.session_state and st.session_state["rtsp_frames"]:
                query_img = Image.fromarray(st.session_state["rtsp_frames"][0])

        if query_img is not None:
            import base64
            from io import BytesIO
            buf = BytesIO()
            query_img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="scan-frame">
                <img src="data:image/jpeg;base64,{b64}" width="350" style="max-width:100%;" />
            </div>
            <div class="scan-label">◉ BIOMETRIC SCAN ACTIVE</div>
            """, unsafe_allow_html=True)
            if st.button("🔍 Recognize & Mark Attendance", type="primary", use_container_width=True):
                with st.spinner("AI is recognizing face..."):
                    img_array = np.array(query_img.convert("RGB"))
                    result = recognize_face(img_array)
                    if result:
                        name, roll, conf = result
                        ok, msg = mark_attendance(name, roll, conf)
                        if ok:
                            st.success(f"✅ {msg}")
                            st.markdown(f'<div class="success-box"><span>🎯</span><span><b>Recognized:</b> {name} ({roll})<br><b>Confidence:</b> {conf:.1f}%</span></div>', unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.warning(msg)
                    else:
                        st.error("Face not recognized. Make sure the person is registered and photo is clear.")

    with col2:
        st.subheader("Today's Live Log")
        today_str = date.today().isoformat()
        today_att = get_attendance(today_str)
        if not today_att.empty:
            for _, row in today_att.iterrows():
                st.markdown(f'<div class="info-box"><span>🟢</span><span><b>{row["name"]}</b> ({row["roll_no"]}) • {row["time"]} • {row["confidence"]:.0f}%</span></div>', unsafe_allow_html=True)
        else:
            st.info("No attendance marked yet today.")

elif page == "📊 Records & Reports":
    hero("ARCHIVE", "Attendance Records", "Filter, review and export every marked entry")
    attendance = get_attendance()

    if attendance.empty:
        st.info("No attendance records yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            dates = sorted(attendance["date"].unique(), reverse=True)
            selected_date = st.selectbox("Filter by Date", ["All"] + list(dates))
        with col2:
            names = ["All"] + sorted(attendance["name"].unique().tolist())
            selected_name = st.selectbox("Filter by Student", names)

        filtered = attendance.copy()
        if selected_date != "All":
            filtered = filtered[filtered["date"] == selected_date]
        if selected_name != "All":
            filtered = filtered[filtered["name"] == selected_name]

        st.dataframe(
            filtered[["name", "roll_no", "date", "time", "confidence", "status"]].rename(
                columns={"name": "Name", "roll_no": "Roll", "date": "Date", "time": "Time", "confidence": "Conf %", "status": "Status"}
            ),
            use_container_width=True, hide_index=True
        )

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Report", csv, f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)

        if not filtered.empty:
            st.subheader("Attendance by Date")
            daily = filtered.groupby("date").size().reset_index(name="count")
            fig = px.line(daily, x="date", y="count", markers=True, color_discrete_sequence=["#00f5a0"])
            fig.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=2, color="#05070d")))
            st.plotly_chart(themed_chart(fig), use_container_width=True)


elif page == "📁 Excel Management":
    hero("DATA OPS", "Excel Management", "Database ko Excel se manage karo — Sir ke liye easy")

    from backend.database import export_students_to_excel, export_attendance_to_excel, import_students_from_excel

    tab1, tab2, tab3 = st.tabs(["📥 Download Excel", "📤 Upload Students", "ℹ️ Help"])

    with tab1:
        st.subheader("Download Data as Excel")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Students List")
            students_df = get_all_students()
            st.write(f"Total Students: **{len(students_df)}**")
            if st.button("Download Students Excel", use_container_width=True):
                data = export_students_to_excel()
                if data:
                    st.download_button(
                        "Click to Save Students.xlsx",
                        data=data,
                        file_name=f"students_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.warning("No students to export")

        with c2:
            st.markdown("#### Attendance Records")
            att_df = get_attendance()
            st.write(f"Total Records: **{len(att_df)}**")
            if st.button("Download Attendance Excel", use_container_width=True):
                data = export_attendance_to_excel()
                if data:
                    st.download_button(
                        "Click to Save Attendance.xlsx",
                        data=data,
                        file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.warning("No attendance records to export")

    with tab2:
        st.subheader("Bulk Add Students from Excel")
        st.info("Excel mein sirf 2 columns hone chahiye: **Name** aur **Roll No**")

        sample = pd.DataFrame({"Name": ["Rahul Sharma", "Priya Patel"], "Roll No": ["CS2025001", "CS2025002"]})
        from io import BytesIO
        sample_output = BytesIO()
        with pd.ExcelWriter(sample_output, engine="openpyxl") as writer:
            sample.to_excel(writer, index=False, sheet_name="Students")
        st.download_button(
            "📄 Download Sample Template",
            data=sample_output.getvalue(),
            file_name="student_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "xls"])
        if uploaded and st.button("Import Students", type="primary"):
            with st.spinner("Importing..."):
                success, errors = import_students_from_excel(uploaded)
                st.success(f"Successfully added: **{success}** students")
                if errors:
                    st.warning("Some rows failed:")
                    for e in errors[:10]:
                        st.write("- " + e)

    with tab3:
        st.markdown("""
        ### Kaise use karein?

        **1. Download Students**
        - Saari registered students ki list Excel mein aa jaayegi
        - Name, Roll No, Registered Date

        **2. Download Attendance**
        - Attendance records Excel mein export
        - Date, Time, Confidence ke saath

        **3. Bulk Upload Students**
        - Sample template download karo
        - Name aur Roll No bharo
        - Upload karke Import dabao
        - Note: Photos baad mein Register page se add kar sakte ho

        Yeh feature Sir ke liye database manage karna bahut easy bana deta hai.
        """)


elif page == "📅 Timetable & Auto":
    hero("SCHEDULER", "Class Timetable & Auto Attendance", "Daily 8 lectures • Auto trigger 5 min before class ends")

    timetable = [
        {"Lecture": 1, "Subject": "Data Structures", "Start": "09:00", "End": "09:50", "Trigger": "09:45"},
        {"Lecture": 2, "Subject": "Database Systems", "Start": "10:00", "End": "10:50", "Trigger": "10:45"},
        {"Lecture": 3, "Subject": "Operating Systems", "Start": "11:00", "End": "11:50", "Trigger": "11:45"},
        {"Lecture": 4, "Subject": "Computer Networks", "Start": "12:00", "End": "12:50", "Trigger": "12:45"},
        {"Lecture": 5, "Subject": "AI / ML", "Start": "14:00", "End": "14:50", "Trigger": "14:45"},
        {"Lecture": 6, "Subject": "Software Engineering", "Start": "15:00", "End": "15:50", "Trigger": "15:45"},
        {"Lecture": 7, "Subject": "Web Technologies", "Start": "16:00", "End": "16:50", "Trigger": "16:45"},
        {"Lecture": 8, "Subject": "Project Lab", "Start": "17:00", "End": "17:50", "Trigger": "17:50"},
    ]

    st.subheader("Today's Schedule (8 Lectures × 50 min)")
    df_tt = pd.DataFrame(timetable)
    st.dataframe(df_tt, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Auto Attendance Logic (Future Ready)")
    st.info("""
    **Kaise kaam karega (jab CCTV + Scheduler lagega):**
    1. Har lecture ke **End time se 5 minute pehle** system trigger hoga
    2. Us classroom ka CCTV stream (RTSP) open hoga
    3. Faces detect + recognize honge
    4. Attendance mark hogi
    5. Excel + (possible ho to) ERP update hoga
    """)

    st.markdown("#### Simulation (Demo)")
    selected = st.selectbox("Select Lecture to Simulate", [f"Lecture {t['Lecture']} - {t['Subject']} (Trigger {t['Trigger']})" for t in timetable])
    if st.button("Simulate Auto Attendance Trigger", type="primary"):
        st.success(f"Triggered for: {selected}")
        st.write("In real system → RTSP camera se frames capture → Face recognition → Attendance mark → Excel/ERP update")
        st.balloons()

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.caption("Yeh timetable abhi static hai. Future mein ERP se sync ho sakta hai.")


elif page == "🚀 Future Scope":
    hero("ROADMAP", "Future Scope & Vision", "College ERP + CCTV Integration Roadmap")

    st.subheader("1. College ERP Integration")
    st.markdown("""
    - ERP se student photos aur details auto sync
    - Attendance directly ERP ke attendance module mein update
    - Web + Mobile app dono mein real-time dikhe
    - Manual Excel upload ki zarurat kam ho jaayegi
    """)

    st.subheader("2. Classroom CCTV Auto Attendance")
    st.markdown("""
    - Har classroom ke IP Camera / CCTV se RTSP connection
    - Class khatam hone se **5 minute pehle** automatic start
    - Daily 8 lectures ke hisaab se scheduler
    - Multiple cameras support
    """)

    st.subheader("3. Proposed Architecture")
    st.code("""
College ERP (Photos + Student Master)
        │
        ▼
Attendance System (Our Project)
  ├── Face Engine (DeepFace)
  ├── RTSP Camera Streams
  ├── Timetable Scheduler
  └── Excel / ERP Updater
        │
        ├──► Excel Daily Report
        └──► ERP Attendance API
    """, language="text")

    st.subheader("4. Benefits")
    st.markdown("""
    - Teachers ko manual attendance nahi leni padegi
    - Proxy attendance kam hoga
    - Real-time dashboard college admin ko milega
    - Data Excel + ERP dono jagah safe rahega
    """)

    st.success("Yeh sab future improvements hain. Current project standalone strong demo ke liye ready hai.")
    st.caption("IBM Project – AI Based Smart Attendance Monitoring System")

elif page == "🤖 AI Insights":
    hero("ANALYST", "AI-Powered Insights", "Groq LLM generates smart summary of attendance patterns")

    students = get_all_students()
    attendance = get_attendance()

    if st.button("✨ Generate AI Report", type="primary"):
        with st.spinner("Asking Groq AI for insights..."):
            insight = get_ai_insight(attendance, students)
            st.markdown(f'<div class="info-box" style="font-size:1.02rem; line-height:1.6;"><span>🧠</span><span>{insight}</span></div>', unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Quick Stats")
    today_str = date.today().isoformat()
    today_count = len(attendance[attendance["date"] == today_str]) if not attendance.empty else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("👥", len(students), "Total Students")
    with c2:
        metric_card("✅", today_count, "Present Today")
    with c3:
        metric_card("🗂️", len(attendance), "Total Records")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<p class='foot-note'>SCANIX · FRONTEND STREAMLIT · BACKEND DATABASE / FACE_ENGINE / AI_INSIGHTS</p>", unsafe_allow_html=True)

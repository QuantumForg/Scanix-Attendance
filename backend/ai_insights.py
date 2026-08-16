import os
from datetime import date

try:
    from groq import Groq
    GROQ_OK = True
except Exception as e:
    GROQ_OK = False
    print("Groq not available:", e)


def get_ai_insight(attendance_df, students_df):
    """Generate attendance insight using Groq LLM. Graceful if no key."""
    if not GROQ_OK:
        return "Groq API not configured. Set GROQ_API_KEY for AI insights."

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "No GROQ_API_KEY found. Add it as environment variable for AI-powered insights."

    try:
        client = Groq(api_key=api_key)

        present_count = 0
        if not attendance_df.empty:
            present_count = len(attendance_df[attendance_df["date"] == date.today().isoformat()])

        total = len(students_df)
        summary = f"Total registered students: {total}. Present today: {present_count}. "
        if not attendance_df.empty:
            summary += f"Recent records: {attendance_df.head(5).to_string()}"

        prompt = f"""You are an AI assistant for a smart attendance system at a college/IBM project.
Based on this data: {summary}
Give a short, professional 3-4 sentence insight report about today's attendance, trends, and one suggestion. Keep it friendly and useful for teachers."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI insight unavailable right now: {str(e)}"

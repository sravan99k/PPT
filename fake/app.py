import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from io import BytesIO
import fpdf
import base64
import tempfile

# --- Function Definitions ---

def classify_risk(percent):
    if percent < 30:
        return "Low"
    elif 30 <= percent < 60:
        return "Moderate"
    else:
        return "High"

def personalized_guidance(row):
    guidance = []
    if row["Stress %"] > 60:
        guidance.append("⚠️ High stress detected. Try relaxation techniques like deep breathing, yoga, or regular physical activity.")
    if row["Depression %"] > 60:
        guidance.append("🧠 Signs of possible depression. You should speak with a mental health professional.")
    if row["Eating Disorder %"] > 60:
        guidance.append("🍽️ Eating behavior concerns found. Practice balanced eating, avoid crash dieting, and consider seeing a nutritionist.")
    if row["Behavioral Issues %"] > 60:
        guidance.append("🧒 Behavioral difficulties noticed. Seek mentorship or conflict resolution workshops.")
    if not guidance:
        return "✅ You're doing well. Keep practicing positive habits!"
    return " ".join(guidance)

def create_student_pdf(student_info, image_bytes):
    pdf = fpdf.FPDF()
    pdf.add_page()
    try:
        font_path = os.path.join(os.path.dirname(__file__), "DejaVuSansCondensed.ttf")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    except:
        pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Student ID: {student_info['SNO']}", ln=True)
    pdf.cell(200, 10, txt=f"Overall Mental Health Status: {student_info['Overall Mental Health Status']}", ln=True)
    pdf.cell(200, 10, txt=f"Mental Health Risk %: {student_info['Mental Health Risk %']:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Risk Class: {student_info['Predicted Risk Class']}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Disorder-wise Risk Percentages:", ln=True)
    for cat in ["Stress", "Depression", "Eating Disorder", "Behavioral Issues"]:
        pdf.cell(200, 10, txt=f"{cat}: {student_info[cat + ' %']:.2f}%", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Suggested Guidance:", ln=True)
    pdf.multi_cell(190, 10, txt=student_info["Suggested Guidance"])

    # Visualization image
    pdf.ln(10)
    pdf.cell(200, 10, txt="Visualization:", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(image_bytes)
        temp_path = tmp_file.name

    try:
        pdf.image(temp_path, x=10, y=pdf.get_y(), w=150)
    except Exception as e:
        print(f"Error loading image: {e}")
    finally:
        os.remove(temp_path)

    output = BytesIO()
    pdf.output(output)
    return output.getvalue()

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧠 Student Mental Health Assessment Dashboard")

# Load CSV
csv_path = os.path.join(os.path.dirname(__file__), "Student_Survey_Responses_300.csv")
df = pd.read_csv(csv_path, encoding="ISO-8859-1")

risk_score_map = {
    "Always": 1.0, "Often": 0.75, "Sometimes": 0.5,
    "Rarely": 0.25, "Never": 0.0, "Not Sure": 0.5,
    "Skip": 0.5, "Yes": 1.0, "No": 0.0
}

# Disorder categories
disorder_categories = {
    "Stress": ["I feel overwhelmed by my emotions", "I often feel anxious", "I often feel lonely or tearful"],
    "Depression": ["I have felt hopeless or helpless recently", "I feel like life is not worth living", "I have thoughts of hurting myself"],
    "Eating Disorder": ["I worry excessively about gaining weight", "I feel pressure to look a certain way because of social media or peers", "I restrict food intake to control my weight", "I skip meals intentionally", "I eat even when I'm not hungry due to stress or emotions", "I feel guilty after eating", "I avoid eating in front of others", "Do you think your eating habits affect your emotional or physical well-being?"],
    "Behavioral Issues": ["I get into fights with my classmates or friends", "I skip school or classes without a good reason", "I tend to lie or hide the truth to avoid trouble", "I have trouble following rules or instructions", "I find it difficult to share my feelings with others"]
}

for disorder, questions in disorder_categories.items():
    df[disorder + " %"] = df[questions].applymap(lambda x: risk_score_map.get(str(x).strip(), 0.5)).mean(axis=1) * 100

df["Mental Health Risk %"] = df[[col for col in df.columns if col.endswith(" %")]].mean(axis=1)
df["Mental Health Category"] = df["Mental Health Risk %"].apply(classify_risk)

le = LabelEncoder()
df["Mental Health Category Encoded"] = le.fit_transform(df["Mental Health Category"])

X = df[[col for col in df.columns if col.endswith(" %") and col != "Mental Health Risk %"]]
y = df["Mental Health Category Encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_train)

df["Predicted Risk Class Encoded"] = clf_model.predict(X)
df["Predicted Risk Class"] = le.inverse_transform(df["Predicted Risk Class Encoded"])
df["Overall Mental Health Status"] = df["Mental Health Risk %"].apply(classify_risk)
df["Suggested Guidance"] = df.apply(personalized_guidance, axis=1)

# Download Full Report
def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button("📥 Download Full Report (All Students)", data=to_csv(df), file_name="Student_MH_Report.csv", mime="text/csv")

# Sidebar
st.sidebar.title("📋 Individual Student Report")
class_col = next((col for col in df.columns if col.lower() in ["class", "grade"]), None)
filtered_df = df

if class_col:
    class_list = sorted(df[class_col].dropna().unique())
    selected_class = st.sidebar.selectbox(f"Select {class_col}", class_list)
    filtered_df = df[df[class_col] == selected_class]

if 'SNO' in filtered_df.columns:
    student_ids = filtered_df['SNO'].unique()
    selected_student = st.sidebar.selectbox("Select Student ID", student_ids)
    student_data = filtered_df[filtered_df['SNO'] == selected_student].iloc[0]

    st.subheader(f"🧾 Student {selected_student} Report")
    if class_col:
        st.write(f"**{class_col}**: {student_data[class_col]}")
    st.write(f"**Overall Mental Health Status**: {student_data['Overall Mental Health Status']}")
    st.write(f"**Mental Health Risk %**: {student_data['Mental Health Risk %']:.2f}%")
    st.write(f"**Predicted Risk Class**: {student_data['Predicted Risk Class']}")
    st.write(f"**Suggested Guidance**: {student_data['Suggested Guidance']}")

    # Compare with class average
    categories = ["Stress", "Depression", "Eating Disorder", "Behavioral Issues"]
    comp_df = pd.DataFrame({
        "Category": categories,
        "Student": [student_data[cat + " %"] for cat in categories],
        "Class Average": [filtered_df[cat + " %"].mean() for cat in categories]
    }).set_index("Category")
    st.markdown("### 📊 Comparison with Class Averages")
    st.dataframe(comp_df.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(categories))
    ax.barh(x, comp_df["Student"], height=0.4, label="Student", color="skyblue")
    ax.barh([i + 0.4 for i in x], comp_df["Class Average"], height=0.4, label="Class Avg", color="orange")
    ax.set_yticks([i + 0.2 for i in x])
    ax.set_yticklabels(categories)
    ax.set_xlabel("Risk %")
    ax.set_title("Student vs Class Average")
    ax.legend()

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    plt.close(fig)
    image_bytes = buffer.getvalue()

    # Download PDF button
    # pdf_bytes = create_student_pdf(student_data, image_bytes)
    # st.download_button("📄 Download Individual PDF Report", data=pdf_bytes,
    #                    file_name=f"Student_{selected_student}_Report.pdf",
    #                    mime="application/pdf")

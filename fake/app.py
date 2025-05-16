import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import io
import fpdf
import os
import base64
from io import BytesIO
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
        pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
        pdf.set_font("DejaVu", size=12)
    except Exception as e:
        pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Student ID: {student_info['SNO']}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Overall Mental Health Status: {student_info['Overall Mental Health Status']}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Mental Health Risk %: {student_info['Mental Health Risk %']:.2f}%", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Predicted Risk Class: {student_info['Predicted Risk Class']}", ln=True, align='L')

    pdf.ln(10)
    pdf.cell(200, 10, text="Disorder-wise Risk Percentages:", ln=True, align='L')
    for category in ["Stress", "Depression", "Eating Disorder", "Behavioral Issues"]:
        pdf.cell(200, 10, text=f"{category}: {student_info[category + ' %']:.2f}%", ln=True, align='L')

    pdf.ln(10)
    pdf.cell(200, 10, text="Suggested Guidance:", ln=True, align='L')
    pdf.multi_cell(190, 10, text=student_info['Suggested Guidance'], align='L')

    pdf.ln(10)
    pdf.cell(200, 10, text="Visualization:", ln=True, align='L')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(image_bytes)
        temp_image_path = tmp_file.name

    try:
        pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=150)
    except Exception as e:
        print(f"Error embedding image: {e}")
    finally:
        os.remove(temp_image_path)

    return pdf.output(dest='S')

# --- Main Streamlit Application ---

st.set_page_config(layout="wide")
st.title("🧠 Student Mental Health Assessment Dashboard")

csv_path = os.path.join(os.path.dirname(__file__), "Student_Survey_Responses_300.csv")
df = pd.read_csv(csv_path, encoding="ISO-8859-1")

risk_score_map = {
    "Always": 1.0, "Often": 0.75, "Sometimes": 0.5,
    "Rarely": 0.25, "Never": 0.0, "Not Sure": 0.5,
    "Skip": 0.5, "Yes": 1.0, "No": 0.0
}

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

# Download full CSV report
def to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv_full_report = to_csv(df)
st.download_button("📥 Download Full Report (All Students)", data=csv_full_report,
                   file_name="Full_Student_Mental_Health_Report.csv", mime="text/csv")

# Individual student section
st.sidebar.title("📋 Individual Student Report")
# --- Add class filter if available ---
# Find class/grade column, case-insensitive
possible_class_cols = [col for col in df.columns if col.lower() in ["class", "grade"]]
class_col = possible_class_cols[0] if possible_class_cols else None

# Sidebar: filter by class if available
if class_col:
    class_list = sorted(df[class_col].dropna().unique())
    selected_class = st.sidebar.selectbox(f"Select {class_col}", class_list)
    filtered_df = df[df[class_col] == selected_class]
else:
    filtered_df = df


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

    # Comparison with class averages
    st.markdown("### 📊 Comparison with Class Averages")
    categories = ["Stress", "Depression", "Eating Disorder", "Behavioral Issues"]
    comparison_df = pd.DataFrame({
        "Category": categories,
        "Student": [student_data[cat + " %"] for cat in categories],
        "Class Average": [filtered_df[cat + " %"].mean() for cat in categories]
    })

    comparison_df.set_index("Category", inplace=True)
    st.dataframe(comparison_df.style.format("{:.2f}"))

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(categories))
    ax.barh(x, comparison_df["Student"], height=0.4, label="Student", color="skyblue")
    ax.barh([i + 0.4 for i in x], comparison_df["Class Average"], height=0.4, label="Class Avg", color="orange")
    ax.set_yticks([i + 0.2 for i in x])
    ax.set_yticklabels(categories)
    ax.set_xlabel("Risk Percentage")
    ax.set_title(f"Student vs Class Average for Student {selected_student}")
    ax.legend()

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    img_bytes = buf.getvalue()

    # Download PDF report
    pdf_bytes = create_student_pdf(student_data, img_bytes)
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="Student_{selected_student}_Report.pdf">📄 Download Individual Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

    # --- Student-Friendly Visualizations ---

    st.markdown("### 🧩 Student-Friendly Visualizations")

    # Pie chart for student's disorder-wise percentages
    pie_labels = categories
    pie_values = [student_data[cat + " %"] for cat in categories]
    pie_colors = sns.color_palette("pastel")[0:4]
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Heatmap of risk levels
    st.markdown("#### 🔥 Risk Heatmap")
    heat_df = pd.DataFrame([pie_values], columns=categories, index=["Student"])
    fig2, ax2 = plt.subplots()
    sns.heatmap(heat_df, annot=True, cmap="Reds", fmt=".1f", ax=ax2)
    st.pyplot(fig2)

    # Bar plot of student's vs top 5 highest average category across class
    st.markdown("#### 📈 Class Risk Trend")
    class_avg = {cat: df[cat + " %"].mean() for cat in categories}
    trend_df = pd.DataFrame({
        "Category": categories,
        "Student": pie_values,
        "Class Avg": list(class_avg.values())
    }).melt(id_vars="Category", var_name="Type", value_name="Percentage")

    fig3, ax3 = plt.subplots()
    sns.barplot(data=trend_df, x="Category", y="Percentage", hue="Type", palette="Set2", ax=ax3)
    ax3.set_title("Comparison of Student vs Class Risk Trends")
    st.pyplot(fig3)

else:
    st.sidebar.warning("The 'SNO' column was not found in the CSV file.")

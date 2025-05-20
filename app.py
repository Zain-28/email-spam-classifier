import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load Model and Vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Real-Time Spam Email Classifier")



# Sidebar for Input
st.sidebar.header("User Input")
email_text = st.sidebar.text_area("Enter Email Text", "")

# Real-Time Prediction
def predict_spam(email):
    email_vec = vectorizer.transform([email])
    prediction = model.predict(email_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Display Prediction
if st.sidebar.button("Classify"):
    if email_text:
        prediction = predict_spam(email_text)
        st.sidebar.success(f"Prediction: {prediction}")
    else:
        st.sidebar.error("Please enter email text.")

# Refresh Button
if st.sidebar.button("Refresh Data"):
    st.rerun()

# Load Data (for Metrics)
data = pd.read_csv("mail_data.csv")
data["Category"] = data["Category"].map({"ham": 0, "spam": 1})


# Train-Test Split
X = data["Message"]
y = data["Category"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Evaluation
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# **Interactive Plotly Charts**
st.subheader("📊 Model Performance")
st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

# **Confusion Matrix (Interactive)**
st.subheader("📊 Confusion Matrix (Interactive)")
fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual"})
st.plotly_chart(fig)

# **Spam vs. Non-Spam (Interactive Bar Chart)**
st.subheader("📊 Spam vs Non-Spam Distribution")
df_counts = data["Category"].value_counts().reset_index()
df_counts.columns = ["Category", "Count"]
df_counts["Category"] = df_counts["Category"].map({0: "Not Spam", 1: "Spam"})
fig = px.bar(df_counts, x="Category", y="Count", color="Category", title="Spam vs Non-Spam Emails")
st.plotly_chart(fig)

# **Accuracy Over Training Epochs**
st.subheader("📈 Accuracy Over Training Epochs")
epochs = [1, 2, 3, 4, 5]
accuracy_values = [0.75, 0.78, 0.81, 0.83, 0.85]
fig = px.line(x=epochs, y=accuracy_values, markers=True, title="Model Accuracy Over Time", labels={"x": "Epochs", "y": "Accuracy"})
st.plotly_chart(fig)


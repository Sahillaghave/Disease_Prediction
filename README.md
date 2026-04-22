# 🧠 AI Disease Predictor

A modern AI-powered web application that predicts diseases based on user-input symptoms using Machine Learning and Django.

---

## 📌 Features

- 🔍 Predict disease from symptoms
- 📊 Confidence score with level (Low / Medium / High)
- 🧠 Top 3 possible disease predictions
- 📖 Disease description
- 💊 Precautions and prevention tips
- 🎨 Modern glassmorphism UI
- ⚡ Smooth scroll user experience

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS (Modern UI)
- **Backend:** Django (Python)
- **ML Model:** Scikit-learn
- **Libraries:** Pandas, NumPy

---

## 🏗️ Project Structure
Disease_Prediction/
│
├── disease_project/ # Django main project
├── predictor/ # App (ML + Views)
│ ├── model.pkl
│ ├── vectorizer.pkl
│ ├── label_encoder.pkl
│ ├── disease_description.csv
│ ├── disease_precaution.csv
│ └── templates/
│
├── manage.py
├── requirements.txt
└── Procfile


---

## ⚙️ How It Works

1. User enters symptoms
2. Text is cleaned and vectorized (TF-IDF)
3. ML model predicts disease
4. Confidence + top 3 predictions generated
5. Disease info & precautions displayed

---

## 🧪 Example Input


fever headache nausea


---

## 📊 Output

- Disease Name
- Confidence %
- Confidence Level
- Description
- Precautions
- Top 3 Predictions

---

## 🚀 Installation (Local Setup)

git clone https://github.com/Sahillaghave/Disease_Prediction.git
cd Disease_Prediction

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python manage.py runserver

## 🌐 Deployment

This project is deployed using Render.

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

## 👨‍💻 Author

Sahil Laghave

💼 Aspiring AI/ML Engineer
🚀 Passionate about building real-world AI projects

## ⭐ Future Improvements
🔄 Real-time prediction (AJAX)
📱 Mobile responsive UI
🔐 User login system
🌍 Multi-language support
🤖 Advanced ML models (BERT, XGBoost)

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!

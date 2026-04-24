import pandas as pd
from django.shortcuts import render
import os

# ================================
# GLOBAL VARIABLES
# ================================
model = None
vectorizer = None
le = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================================
# TRAIN MODEL ON STARTUP
# ================================
def train_model():
    global model, vectorizer, le

    print("🔥 Training model on server startup...")

    df = pd.read_csv(os.path.join(BASE_DIR, 'predictor/Training.csv'))

    X = df.drop('prognosis', axis=1)
    y = df['prognosis']

    # Convert symptoms to text
    def convert_to_text(row):
        return ' '.join(row.index[row == 1])

    X['Symptoms'] = X.apply(convert_to_text, axis=1)
    X = X['Symptoms']

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # Model
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("✅ Model trained successfully!")
    print("Fitted:", hasattr(vectorizer, "idf_"))

# 🔥 CALL TRAINING HERE
train_model()

# ================================
# LOAD EXTRA DATA
# ================================
desc = pd.read_csv(os.path.join(BASE_DIR, 'predictor/disease_description.csv'))
prec = pd.read_csv(os.path.join(BASE_DIR, 'predictor/disease_precaution.csv'))

desc_dict = dict(zip(desc.iloc[:, 0], desc.iloc[:, 1]))

prec_dict = {}
for _, row in prec.iterrows():
    disease = row.iloc[0]
    precautions = [x for x in row.iloc[1:] if pd.notna(x)]
    prec_dict[disease] = precautions

# ================================
# HELPERS
# ================================
def clean_input(text):
    return text.lower().replace(',', ' ')

def get_confidence_level(conf):
    if conf > 0.6:
        return "High"
    elif conf > 0.3:
        return "Medium"
    else:
        return "Low"

# ================================
# PREDICTION FUNCTION
# ================================
def predict_disease(text):
    text_vec = vectorizer.transform([text])

    pred = model.predict(text_vec)
    disease = le.inverse_transform(pred)[0]

    probs = model.predict_proba(text_vec)[0]
    confidence = max(probs)

    confidence_percent = round(confidence * 100, 2)
    confidence_level = get_confidence_level(confidence)

    # Top 3 predictions
    top3_idx = probs.argsort()[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)

    description = desc_dict.get(disease, "No description available")
    precautions = prec_dict.get(disease, [])

    return disease, description, precautions, confidence_percent, confidence_level, top3_diseases

# ================================
# MAIN VIEW
# ================================
def home(request):
    result = None
    description = None
    precautions = None
    confidence = None
    confidence_level = None
    top3 = None
    error = None

    if request.method == 'POST':
        symptoms = request.POST.get('symptoms')

        if not symptoms or not symptoms.strip():
            error = "⚠️ Please enter symptoms"
        else:
            symptoms = clean_input(symptoms)

            if len(symptoms.split()) < 3:
                error = "⚠️ Enter at least 3 symptoms for better prediction"
            else:
                result, description, precautions, confidence, confidence_level, top3 = predict_disease(symptoms)

    return render(request, 'index.html', {
        'result': result,
        'description': description,
        'precautions': precautions,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'top3': top3,
        'error': error
    })
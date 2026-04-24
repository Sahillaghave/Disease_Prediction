import pickle
import pandas as pd
from django.shortcuts import render
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load pipeline
pipeline = pickle.load(open(os.path.join(BASE_DIR, 'predictor/pipeline.pkl'), 'rb'))
le = pickle.load(open(os.path.join(BASE_DIR, 'predictor/label_encoder.pkl'), 'rb'))

# Load CSVs
desc = pd.read_csv(os.path.join(BASE_DIR, 'predictor/disease_description.csv'))
prec = pd.read_csv(os.path.join(BASE_DIR, 'predictor/disease_precaution.csv'))

# Create dictionaries
desc_dict = dict(zip(desc.iloc[:, 0], desc.iloc[:, 1]))

prec_dict = {}
for _, row in prec.iterrows():
    disease = row.iloc[0]
    precautions = [x for x in row.iloc[1:] if pd.notna(x)]
    prec_dict[disease] = precautions

# Clean input
def clean_input(text):
    return text.lower().replace(',', ' ')

# Confidence level
def get_confidence_level(conf):
    if conf > 0.6:
        return "High"
    elif conf > 0.3:
        return "Medium"
    else:
        return "Low"

# Prediction function
def predict_disease(text):
    pred = pipeline.predict([text])
    disease = le.inverse_transform(pred)[0]

    probs = pipeline.predict_proba([text])[0]
    confidence = max(probs)

    confidence_percent = round(confidence * 100, 2)
    confidence_level = get_confidence_level(confidence)

    top3_idx = probs.argsort()[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)

    description = desc_dict.get(disease, "No description available")
    precautions = prec_dict.get(disease, [])

    return disease, description, precautions, confidence_percent, confidence_level, top3_diseases

# View
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
                error = "⚠️ Enter at least 3 symptoms"
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
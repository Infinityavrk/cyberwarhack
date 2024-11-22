from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
import nltk
import re

app = Flask(__name__)

# Download required NLTK data
nltk.download(['punkt', 'stopwords'])

# Load model and components
def load_model():
    model_path = 'cybercrime_model/'
    components = joblib.load(f'{model_path}/model_components.joblib')
    return (
        components['category_model'],
        components['subcategory_model'],
        components['tfidf'],
        components['category_encoder'],
        components['subcategory_encoder']
    )

def preprocess_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [t for t in word_tokenize(text) if t not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.json['text']
        
        # Load models and preprocessors
        category_model, subcategory_model, tfidf, category_encoder, subcategory_encoder = load_model()
        stop_words = set(stopwords.words('english'))
        
        # Preprocess input
        processed_text = preprocess_text(text, stop_words)
        X_tfidf = tfidf.transform([processed_text])
        
        # Make predictions
        cat_pred = category_model.predict(X_tfidf)[0]
        subcat_pred = subcategory_model.predict(X_tfidf)[0]
        
        # Get predicted categories
        category = category_encoder.inverse_transform([cat_pred])[0]
        subcategory = subcategory_encoder.inverse_transform([subcat_pred])[0]
        
        # Get confidence scores
        cat_confidence = process.extractOne(category, category_encoder.classes_)[1]
        subcat_confidence = process.extractOne(subcategory, subcategory_encoder.classes_)[1]
        
        return jsonify({
            'success': True,
            'prediction': {
                'category': category,
                'subcategory': subcategory,
                'category_confidence': cat_confidence,
                'subcategory_confidence': subcat_confidence
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
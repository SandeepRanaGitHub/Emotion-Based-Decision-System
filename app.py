
import flask
from flask import Flask,  request, jsonify
import joblib

app = Flask(__name__)
# Load The MODELs
model_state = joblib.load('model_state.joblib')
model_intensity = joblib.load('model_intensity.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')


# NLP SetUp
wordnet = WordNetLemmatizer()
stopword = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [wordnet.lemmatize(word) for word in tokens if word not in stopword]
    return ' '.join(tokens)

 # Decision Layer
def decision_layer(state, intensity):
   intensity = round(float(intensity))

   if state in ['stress', 'anxiety']:
     return "breathing exercise", "now"
   elif state in ["sad"]:
     return "talk to someone", "tonight"
   elif state in ["happy"]:
     return "continue routine", "now"
   else:
     return "reflect", "later_today"


 # API Route
@app.route("/")
def home():
    return "Emotion-Based Decision API is running 🚀"

@app.route('/predict',methods = ['POST'])
def predict():
   data = request.get_json(force=True)
   # Convert data into DataFrame
   df = pd.DataFrame([data])

   # Preprocess text
   df['clean_text'] = df['journal_text'].apply(preprocess_text)
   #TF-IDF
   X_text = vectorizer.transform(df['clean_text'])

   # REMOVE TEXT COLUMNS
   df = df.drop(columns=["journal_text", "clean_text"], errors='ignore')

   # One-hot Encode
   df = pd.get_dummies(df)
   # Align Columns
   df = df.reindex(columns = feature_columns, fill_value=0)

   #Scale numeric
   num_cols = ['duration_min','sleep_hours','energy_level','stress_level']
   df[num_cols] = scaler.transform(df[num_cols])

   # Combine
   X = hstack((X_text, df.astype(float)))
   # Truncated SVD
   X = svd.transform(X)

   # Predictions
   state = model_state.predict(X)[0]
   intensity = float(model_intensity.predict(X)[0])

   # Confidence
   probs = model_state.predict_proba(X)
   confidence = float(np.max(probs))

   uncertain_flag = int(confidence<0.6)

   # Decision
   action, timing = decision_layer(state, intensity)

   return jsonify({
         "predicted_state": state,
         "predicted_intensity": round(intensity, 2),
         "confidence": round(confidence, 3),
         "uncertain_flag": uncertain_flag,
         "what_to_do": action,
         "when_to_do": timing
     })

if __name__ == "__main__":
    app.run(host = 0.0.0.0, port = 5000)


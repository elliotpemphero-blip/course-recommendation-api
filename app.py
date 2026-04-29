from flask import Flask, request, jsonify
from flask_cors import CORS  # 1. ADDED THIS
import joblib
import pandas as pd
import requests

app = Flask(__name__)
CORS(app)  # 2. ENABLED CORS so your frontend can connect

# Load ML model
model = joblib.load("course_recommendation_model.pkl")

# Supabase credentials
SUPABASE_URL = "https://snhhlgsjzsmxoocmphjz.supabase.co"
SUPABASE_KEY = "sb_publishable_kVh5f5LQWaJne2SG_1-NAg_vCPmhXll"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        
        # 3. Use .get() to prevent the app from crashing if a subject is missing
        features = pd.DataFrame([[
            data.get("mathematics", 0),
            data.get("english", 0),
            data.get("biology", 0),
            data.get("physics", 0),
            data.get("chemistry", 0),
            data.get("geography", 0),
            data.get("history", 0),
            data.get("agriculture", 0),
            data.get("business_studies", 0),
            data.get("chichewa", 0)
        ]], columns=[
            "Mathematics", "English", "Biology", "Physics", "Chemistry",
            "Geography", "History", "Agriculture", "BusinessStudies", "Chichewa"
        ])

        # Predict field
        predicted_field = model.predict(features)[0]

        # Fetch from Supabase
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/programs?field=eq.{predicted_field}",
            headers=headers
        )
        
        programs = response.json()

        # 4. Added a check to handle empty results or errors from Supabase
        if not isinstance(programs, list):
            return jsonify({"error": "Failed to fetch from database"}), 500

        top_programs = programs[:3]

        return jsonify({
            "predicted_field": predicted_field,
            "programs": top_programs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses environment variables for ports, but 10000 is fine for local
    app.run(host="0.0.0.0", port=10000)
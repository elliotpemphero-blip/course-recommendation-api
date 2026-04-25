from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests

app = Flask(__name__)

# Load ML model
# Ensure 'course_recommendation.pkl' is in the same folder as this script
model = joblib.load("course_recommendation.pkl")

# Your specific Supabase credentials
SUPABASE_URL = "https://snhhlgsjzsmxoocmphjz.supabase.co"
SUPABASE_KEY = "sb_publishable_kVh5f5LQWaJne2SG_1-NAg_vCPmhXll"

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    # Extract all subjects from the incoming JSON
    math = data["mathematics"]
    english = data["english"]
    biology = data["biology"]
    physics = data["physics"]
    chemistry = data["chemistry"]
    geography = data["geography"]
    history = data["history"]
    agriculture = data["agriculture"]
    business = data["business_studies"]
    chichewa = data["chichewa"]

    # Create feature dataframe for the ML model
    features = pd.DataFrame([[
        math,
        english,
        biology,
        physics,
        chemistry,
        geography,
        history,
        agriculture,
        business,
        chichewa
    ]], columns=[
        "Mathematics",
        "English",
        "Biology",
        "Physics",
        "Chemistry",
        "Geography",
        "History",
        "Agriculture",
        "BusinessStudies",
        "Chichewa"
    ])

    # Predict the career field
    predicted_field = model.predict(features)[0]

    # Set up headers for Supabase API authentication
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    # Fetch programs from your Supabase 'programs' table based on the predicted field
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/programs?field=eq.{predicted_field}",
        headers=headers
    )

    programs = response.json()

    # Get the top 3 program results
    top_programs = programs[:3]

    # Return the results back to your website frontend
    return jsonify({
        "predicted_field": predicted_field,
        "programs": top_programs
    })

if __name__ == "__main__":
    # Runs the server on port 10000
    app.run(host="0.0.0.0", port=10000)
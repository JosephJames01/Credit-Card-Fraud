from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

print("Loading models...")
model = joblib.load('c:/Users/Josep/Downloads/Fundementals of Financial Technology-20260213T142952Z-1-001/Fundementals of Financial Technology/credit/fraud_mlp_model.joblib')
scaler = joblib.load('c:/Users/Josep/Downloads/Fundementals of Financial Technology-20260213T142952Z-1-001/Fundementals of Financial Technology/credit/fraud_scaler.joblib')
print("✅ Models loaded successfully!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    
    if request.method == 'POST':
        try:
            # 29 features: V1-V28 + Amount
            data = [float(request.form.get(f'feature_{i}', 0)) for i in range(29)]
            df = pd.DataFrame([data])
            df_scaled = scaler.transform(df)
            prob = model.predict_proba(df_scaled)[0, 1]
            pred = 1 if prob > 0.5 else 0
            prediction = 'FRAUD ⚠️' if pred else 'LEGIT ✅'
            probability = f"{prob:.1%}"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    html = """
<!DOCTYPE html>
<html>
<head><title>🛡️ Fraud Detector</title>
<style>
body{font-family:Arial;max-width:900px;margin:auto;padding:20px;background:#667eea;color:#333}
.container{background:white;padding:40px;border-radius:20px}
h1{text-align:center;color:#333}
input{width:100%;padding:10px;margin:5px 0;border:2px solid #ddd;border-radius:8px}
button{width:100%;padding:15px;background:#667eea;color:white;border:none;border-radius:10px;font-size:18px;cursor:pointer}
button:hover{background:#5568d3}
.result{margin-top:20px;padding:20px;border-radius:10px;text-align:center;font-size:1.3em;font-weight:bold}
.fraud{background:#ff6b6b;color:white}
.legit{background:#51cf66;color:white}
.features{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
label{font-size:0.9em;color:#555;font-weight:bold}
</style>
</head>
<body>
<div class="container">
<h1>🛡️ Credit Card Fraud Detector</h1>
<p style="text-align:center;color:#666;margin-bottom:30px">29 Features: V1-V28 + Amount (F1=0.77)</p>
<form method="POST">
<div class="features">
{% for i in range(0, 28) %}
<div><label>V{{i+1}}:</label><input type="number" step="0.001" name="feature_{{i}}" value="0" required></div>
{% endfor %}
<div><label>Amount:</label><input type="number" step="0.01" name="feature_28" value="149.62" required></div>
</div>
<button type="submit">🔍 Detect Fraud</button>
</form>
{% if prediction %}
<div class="result {{ 'fraud' if 'FRAUD' in prediction else 'legit' }}">
{{ prediction }}<br>Probability: {{ probability }}
</div>
{% endif %}
</div>
</body>
</html>
"""
    return render_template_string(html, prediction=prediction, probability=probability)

if __name__ == '__main__':
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)

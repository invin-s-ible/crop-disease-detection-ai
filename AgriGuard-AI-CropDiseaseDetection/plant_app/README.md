# AgriGuard AI — Crop Disease Detection & Plant Health Monitoring

A solo minor project: an AI-powered web app that identifies crop diseases from leaf
photos and predicts plant health from environmental sensor readings.

## Features

- **AI Leaf Disease Detection** — upload a photo of a plant leaf; a vision-language
  model (via Groq) identifies the disease, severity, symptoms, causes, and
  treatment/prevention recommendations.
- **Sensor-Based Health Prediction** — train two ML classifiers (Random Forest & SVM)
  on your own temperature/humidity/soil-moisture/light-intensity data, then predict
  plant health from new readings.
- **Disease Library** — a quick reference of common crop diseases, causes and
  prevention tips.
- **Dashboard analytics** — live charts of scan outcomes and detected disease types.
- **Accounts & history** — register/login, and a full log of every scan and test.

## Tech Stack

Python, Flask, Flask-SQLAlchemy (SQLite), pandas, NumPy, scikit-learn, Groq API
(vision model), Bootstrap 5, Chart.js.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**
   Copy `.env.example` to `.env` (if you don't already have a `.env`) and set your
   own Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Get a free key at https://console.groq.com/keys — also change `SECRET_KEY` to a
   random string before deploying anywhere public.

3. **Run the app**
   - Windows: double-click `run.bat`
   - macOS/Linux: `bash run.sh`
   - Or directly: `python app.py`

   The app runs at http://localhost:5000

4. **(Optional) Generate sample data** to test the sensor-training flow:
   ```bash
   python generate_test_data.py
   ```

## Project Structure

```
app.py                  Flask app: models, routes, ML training, Groq vision integration
templates/               All Jinja2 HTML pages
static/css/style.css     App styling
static/js/main.js        Drag-and-drop upload + AJAX analysis + chart wiring
generate_test_data.py    Sample sensor/image data generator
sample_data.csv          Sample training data
requirements.txt         Python dependencies
run.bat / run.sh         One-click startup scripts
```

## Notes for submission

- **Deliverables mapping**: source code (this repo), dataset details (`sample_data.csv`,
  `generate_test_data.py`), trained model (generated at `models_data/*.pkl` after you
  train via the "Train Model" page), working application (this Flask app), model
  evaluation report (shown on the Training Results page: accuracy/precision/recall/F1
  for both models), demo video (record yourself walking through: register → upload a
  leaf photo → view AI diagnosis → train a model → run a sensor prediction).
- The image-based detector uses a hosted vision-language model (Groq) rather than a
  locally trained CNN, since no labeled leaf-disease image dataset was provided. If
  your project brief specifically requires a *locally trained* image classifier (e.g.
  with TensorFlow/PyTorch on a dataset like PlantVillage), let me know and I can help
  you build and swap that in.

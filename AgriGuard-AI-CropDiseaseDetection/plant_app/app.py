from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

import os
import json
import logging
import sys
import base64
import csv
import io
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import joblib

from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_plant_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models_data', exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_image_file(filename: str) -> bool:
    """Check whether an uploaded filename has an allowed image extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ============================================================================
# CROP DISEASE LIBRARY (reference data for the Disease Library feature)
# ============================================================================

DISEASE_LIBRARY = [
    {
        'name': 'Powdery Mildew', 'type': 'Fungal', 'crops': 'Wheat, grapes, cucurbits, tomatoes',
        'symptoms': ['White/grey powdery spots on leaves', 'Leaf curling and yellowing', 'Stunted growth'],
        'causes': ['High humidity with dry leaf surfaces', 'Poor air circulation', 'Warm days & cool nights'],
        'prevention': ['Space plants for airflow', 'Avoid overhead watering', 'Apply sulfur or neem-based fungicide early']
    },
    {
        'name': 'Leaf Rust', 'type': 'Fungal', 'crops': 'Wheat, maize, coffee, beans',
        'symptoms': ['Orange-brown pustules on leaf surface', 'Yellowing around spots', 'Premature leaf drop'],
        'causes': ['Wet foliage for extended periods', 'High humidity', 'Spread by wind-borne spores'],
        'prevention': ['Plant rust-resistant varieties', 'Remove infected debris', 'Apply fungicide at first sign']
    },
    {
        'name': 'Bacterial Leaf Blight', 'type': 'Bacterial', 'crops': 'Rice, cotton, beans',
        'symptoms': ['Water-soaked lesions turning yellow-brown', 'Wilting of leaves', 'Streaks along veins'],
        'causes': ['Contaminated seed or water', 'Wounds from insects or wind', 'Warm, humid conditions'],
        'prevention': ['Use certified disease-free seed', 'Avoid field work when leaves are wet', 'Practice crop rotation']
    },
    {
        'name': 'Mosaic Virus', 'type': 'Viral', 'crops': 'Tomato, cucumber, tobacco, papaya',
        'symptoms': ['Mottled light/dark green patterns', 'Leaf distortion and curling', 'Stunted plant growth'],
        'causes': ['Transmitted by aphids/whiteflies', 'Contaminated tools', 'Infected seedlings'],
        'prevention': ['Control insect vectors', 'Disinfect tools between plants', 'Remove and destroy infected plants']
    },
    {
        'name': 'Spider Mite Infestation', 'type': 'Pest', 'crops': 'Most vegetable & fruit crops',
        'symptoms': ['Fine webbing under leaves', 'Tiny yellow/white speckles', 'Bronzing of foliage'],
        'causes': ['Hot, dry weather', 'Dust accumulation on leaves', 'Overuse of broad-spectrum pesticides'],
        'prevention': ['Increase humidity around plants', 'Introduce predatory mites', 'Rinse leaves regularly']
    },
    {
        'name': 'Nutrient Deficiency (Nitrogen)', 'type': 'Nutrient deficiency', 'crops': 'All crops',
        'symptoms': ['Pale yellow older leaves', 'Slow, stunted growth', 'Thin, weak stems'],
        'causes': ['Poor soil fertility', 'Excess rainfall leaching nutrients', 'Imbalanced fertilization'],
        'prevention': ['Apply balanced NPK fertilizer', 'Test soil regularly', 'Add organic compost']
    },
]

# Initialize database
db = SQLAlchemy(app)

# ============================================================================
# JINJA2 CUSTOM FILTERS
# ============================================================================

@app.template_filter('fromjson')
def fromjson_filter(value):
    """Convert JSON string to Python object"""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


@app.template_filter('from_json')
def from_json_filter(value):
    """Convert JSON string to Python object (with underscore)"""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


@app.template_filter('slice_list')
def slice_list_filter(value, limit):
    """Slice a list to a certain limit"""
    if not value:
        return []
    return list(value)[:int(limit)]

# ============================================================================
# MODELS
# ============================================================================

class User(db.Model):
    """User model for registration and login"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    training_data = db.relationship('TrainingData', backref='user', lazy=True, cascade='all, delete-orphan')
    test_results = db.relationship('TestResults', backref='user', lazy=True, cascade='all, delete-orphan')
    image_analysis = db.relationship('ImageAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)


class TrainingData(db.Model):
    """Model to store training sessions and trained models"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    model1_name = db.Column(db.String(100), default='Random Forest')
    model1_accuracy = db.Column(db.Float)
    model1_precision = db.Column(db.Float)
    model1_recall = db.Column(db.Float)
    model1_f1 = db.Column(db.Float)
    model1_path = db.Column(db.String(200))
    
    model2_name = db.Column(db.String(100), default='Support Vector Machine')
    model2_accuracy = db.Column(db.Float)
    model2_precision = db.Column(db.Float)
    model2_recall = db.Column(db.Float)
    model2_f1 = db.Column(db.Float)
    model2_path = db.Column(db.String(200))
    
    scaler_path = db.Column(db.String(200))
    feature_names = db.Column(db.String(500))  # JSON string of feature names
    rows_processed = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.now)


class TestResults(db.Model):
    """Model to store sensor data test results"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    soil_moisture = db.Column(db.Float, nullable=False)
    light_intensity = db.Column(db.Float, nullable=False)
    model1_prediction = db.Column(db.String(100))
    model1_confidence = db.Column(db.Float)
    model2_prediction = db.Column(db.String(100))
    model2_confidence = db.Column(db.Float)
    is_healthy = db.Column(db.Boolean)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)


class ImageAnalysis(db.Model):
    """Model to store image analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    disease_detected = db.Column(db.Boolean)
    disease_name = db.Column(db.String(100))
    disease_type = db.Column(db.String(100))
    severity = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    symptoms = db.Column(db.String(500))  # JSON string
    possible_causes = db.Column(db.String(500))  # JSON string
    treatment = db.Column(db.String(500))  # JSON string
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)


# ============================================================================
# LEAF DISEASE DETECTOR (from previous code)
# ============================================================================

@dataclass
class DiseaseAnalysisResult:
    """Data class for storing disease analysis results"""
    disease_detected: bool
    disease_name: Optional[str]
    disease_type: str
    severity: str
    confidence: float
    symptoms: List[str]
    possible_causes: List[str]
    treatment: List[str]
    analysis_timestamp: str = None
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().astimezone().isoformat()


class LeafDiseaseDetector:
    """Leaf Disease Detection System using Groq API"""
    
    MODEL_NAME = "qwen/qwen3.8-27b"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 1024
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Leaf Disease Detector"""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key, timeout=90.0, max_retries=2)
        logger.info("Leaf Disease Detector initialized")
    
    def create_analysis_prompt(self) -> str:
        """Create the analysis prompt"""
        return """IMPORTANT: First determine if this image contains a plant leaf or vegetation. If the image shows humans, animals, objects, buildings, or anything other than plant leaves/vegetation, return the "invalid_image" response format below.

If this is a valid leaf/plant image, analyze it for diseases and return the results in JSON format.

Please identify:
1. Whether this is actually a leaf/plant image
2. Disease name (if any)
3. Disease type/category
4. Severity level (mild, moderate, severe)
5. Confidence score (0-100%)
6. Symptoms observed
7. Possible causes
8. Treatment recommendations

For NON-LEAF images (humans, animals, objects, or not detected as leaves, etc.), return this format:
{
    "disease_detected": false,
    "disease_name": null,
    "disease_type": "invalid_image",
    "severity": "none",
    "confidence": 95,
    "symptoms": ["This image does not contain a plant leaf"],
    "possible_causes": ["Invalid image type uploaded"],
    "treatment": ["Please upload an image of a plant leaf for disease analysis"]
}

For VALID LEAF images, return this format:
{
    "disease_detected": true/false,
    "disease_name": "name of disease or null",
    "disease_type": "fungal/bacterial/viral/pest/nutrient deficiency/healthy",
    "severity": "mild/moderate/severe/none",
    "confidence": 85,
    "symptoms": ["list", "of", "symptoms"],
    "possible_causes": ["list", "of", "causes"],
    "treatment": ["list", "of", "treatments"]
}"""
    
    def analyze_leaf_image_base64(self, base64_image: str,
                                   temperature: float = None,
                                   max_tokens: int = None) -> Dict:
        """Analyze base64 encoded image data for leaf diseases"""
        try:
            logger.info("Starting analysis for base64 image data")
            
            # Validate base64 input
            if not isinstance(base64_image, str):
                raise ValueError("base64_image must be a string")
            
            if not base64_image:
                raise ValueError("base64_image cannot be empty")
            
            # Clean base64 string
            if base64_image.startswith('data:'):
                base64_image = base64_image.split(',', 1)[1]
            
            temperature = temperature or self.DEFAULT_TEMPERATURE
            max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
            
            # Make API request
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.create_analysis_prompt()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            logger.info("API request completed successfully")
            result = self._parse_response(completion.choices[0].message.content)
            return asdict(result)
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Coerce a confidence-like value (which may arrive as '85%', '85', or 85) to a float safely"""
        try:
            if isinstance(value, str):
                value = value.strip().replace('%', '')
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_response(self, response_content: str) -> DiseaseAnalysisResult:
        """Parse and validate API response"""
        try:
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.replace('```', '').strip()
            
            disease_data = json.loads(cleaned_response)
            logger.info("Response parsed successfully as JSON")
            
            return DiseaseAnalysisResult(
                disease_detected=bool(disease_data.get('disease_detected', False)),
                disease_name=disease_data.get('disease_name'),
                disease_type=disease_data.get('disease_type', 'unknown'),
                severity=disease_data.get('severity', 'unknown'),
                confidence=self._safe_float(disease_data.get('confidence', 0)),
                symptoms=disease_data.get('symptoms', []),
                possible_causes=disease_data.get('possible_causes', []),
                treatment=disease_data.get('treatment', [])
            )
        
        except json.JSONDecodeError:
            logger.warning("Failed to parse as JSON, attempting to extract JSON from response")
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    disease_data = json.loads(json_match.group())
                    logger.info("JSON extracted and parsed successfully")
                    
                    return DiseaseAnalysisResult(
                        disease_detected=bool(disease_data.get('disease_detected', False)),
                        disease_name=disease_data.get('disease_name'),
                        disease_type=disease_data.get('disease_type', 'unknown'),
                        severity=disease_data.get('severity', 'unknown'),
                        confidence=self._safe_float(disease_data.get('confidence', 0)),
                        symptoms=disease_data.get('symptoms', []),
                        possible_causes=disease_data.get('possible_causes', []),
                        treatment=disease_data.get('treatment', [])
                    )
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Could not parse response as JSON. Raw response: {response_content}")
            # Degrade gracefully instead of failing the whole request: show an
            # "inconclusive" result rather than a scary error box.
            return DiseaseAnalysisResult(
                disease_detected=False,
                disease_name=None,
                disease_type='inconclusive',
                severity='none',
                confidence=0.0,
                symptoms=['The AI could not produce a clear reading for this image.'],
                possible_causes=['The photo may be too blurry, dark, or not a clear close-up of a leaf.'],
                treatment=['Try again with a sharp, well-lit, close-up photo of a single leaf.']
            )


# ============================================================================
# ML TRAINING UTILITIES
# ============================================================================

class PlantHealthMLTrainer:
    """Machine Learning trainer for plant health prediction"""
    
    @staticmethod
    def load_and_preprocess_csv(filepath: str):
        """Load CSV and preprocess data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Identify features and target (assuming last column is target)
            # Adapt based on your CSV structure
            if 'health_status' in df.columns:
                target_col = 'health_status'
            elif 'label' in df.columns:
                target_col = 'label'
            else:
                target_col = df.columns[-1]
            
            # Select numeric columns for features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            X = df[numeric_cols]
            y = df[target_col]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            return X, y, numeric_cols, target_col
        
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    @staticmethod
    def train_models(X, y, user_id: int):
        """Train two ML models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Model 1: Random Forest
            model1 = RandomForestClassifier(n_estimators=100, random_state=42)
            model1.fit(X_train_scaled, y_train)
            y_pred1 = model1.predict(X_test_scaled)
            
            # Model 1 metrics
            metrics1 = {
                'accuracy': accuracy_score(y_test, y_pred1),
                'precision': precision_score(y_test, y_pred1, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred1, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred1, average='weighted', zero_division=0)
            }
            
            # Train Model 2: SVM
            model2 = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
            model2.fit(X_train_scaled, y_train)
            y_pred2 = model2.predict(X_test_scaled)
            
            # Model 2 metrics
            metrics2 = {
                'accuracy': accuracy_score(y_test, y_pred2),
                'precision': precision_score(y_test, y_pred2, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred2, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred2, average='weighted', zero_division=0)
            }
            
            # Save models
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model1_path = f"models_data/model1_user{user_id}_{timestamp}.pkl"
            model2_path = f"models_data/model2_user{user_id}_{timestamp}.pkl"
            scaler_path = f"models_data/scaler_user{user_id}_{timestamp}.pkl"
            
            joblib.dump(model1, model1_path)
            joblib.dump(model2, model2_path)
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"Models trained and saved for user {user_id}")
            
            return metrics1, metrics2, model1_path, model2_path, scaler_path
        
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise


# ============================================================================
# AUTHENTICATION DECORATORS
# ============================================================================

def login_required(f):
    """Decorator to require login, and to require that the logged-in user still exists"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))

        # Guard against a stale session pointing to a user_id that no longer
        # exists (e.g. the database was reset while a login cookie persisted).
        if User.query.get(session['user_id']) is None:
            session.clear()
            flash('Your session has expired. Please log in again.', 'warning')
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function


# ============================================================================
# ROUTES - AUTHENTICATION
# ============================================================================

@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, else to login"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        phone = request.form.get('phone')
        location = request.form.get('location')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, full_name, phone, location, password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))
        
        # Create user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            phone=phone,
            location=location
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome, {user.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))


# ============================================================================
# ROUTES - MAIN APPLICATION
# ============================================================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard - system overview"""
    user = User.query.get(session['user_id'])
    
    # Get statistics
    total_tests = TestResults.query.filter_by(user_id=user.id).count()
    total_images = ImageAnalysis.query.filter_by(user_id=user.id).count()
    healthy_results = TestResults.query.filter_by(user_id=user.id, is_healthy=True).count()
    trained_models = TrainingData.query.filter_by(user_id=user.id).count()
    
    # Get recent tests
    recent_tests = TestResults.query.filter_by(user_id=user.id).order_by(
        TestResults.created_at.desc()).limit(5).all()
    recent_images = ImageAnalysis.query.filter_by(user_id=user.id).order_by(
        ImageAnalysis.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html',
                          user=user,
                          total_tests=total_tests,
                          total_images=total_images,
                          healthy_results=healthy_results,
                          trained_models=trained_models,
                          recent_tests=recent_tests,
                          recent_images=recent_images)


@app.route('/profile')
@login_required
def profile():
    """User profile"""
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)


@app.route('/train', methods=['GET', 'POST'])
@login_required
def train_model():
    """Train ML models from CSV data"""
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file uploaded.', 'danger')
            return redirect(url_for('train_model'))
        
        file = request.files['csv_file']
        
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('train_model'))
        
        if not file.filename.endswith('.csv'):
            flash('Only CSV files are allowed.', 'danger')
            return redirect(url_for('train_model'))
        
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess data
            X, y, feature_names, target_col = PlantHealthMLTrainer.load_and_preprocess_csv(filepath)
            
            # Train models
            metrics1, metrics2, model1_path, model2_path, scaler_path = PlantHealthMLTrainer.train_models(
                X, y, session['user_id']
            )
            
            # Save training record to database
            training_record = TrainingData(
                user_id=session['user_id'],
                filename=filename,
                model1_name='Random Forest',
                model1_accuracy=metrics1['accuracy'],
                model1_precision=metrics1['precision'],
                model1_recall=metrics1['recall'],
                model1_f1=metrics1['f1'],
                model1_path=model1_path,
                model2_name='Support Vector Machine',
                model2_accuracy=metrics2['accuracy'],
                model2_precision=metrics2['precision'],
                model2_recall=metrics2['recall'],
                model2_f1=metrics2['f1'],
                model2_path=model2_path,
                scaler_path=scaler_path,
                feature_names=json.dumps(feature_names),
                rows_processed=len(X)
            )
            
            db.session.add(training_record)
            db.session.commit()
            
            flash('Models trained successfully!', 'success')
            return redirect(url_for('training_results', training_id=training_record.id))
        
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            flash(f'Training failed: {str(e)}', 'danger')
            return redirect(url_for('train_model'))
    
    user = User.query.get(session.get('user_id'))
    return render_template('train.html', user=user)


@app.route('/training-results/<int:training_id>')
@login_required
def training_results(training_id):
    """View training results"""
    training = TrainingData.query.get_or_404(training_id)
    
    if training.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('training_results.html', training=training)


@app.route('/test-sensors', methods=['GET', 'POST'])
@login_required
def test_sensors():
    """Test sensor data"""
    if request.method == 'POST':
        try:
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            soil_moisture = float(request.form.get('soil_moisture'))
            light_intensity = float(request.form.get('light_intensity'))
            notes = request.form.get('notes', '')
            
            # Get latest trained models
            training = TrainingData.query.filter_by(
                user_id=session['user_id']
            ).order_by(TrainingData.created_at.desc()).first()
            
            if not training:
                flash('No trained models found. Please train a model first.', 'warning')
                return redirect(url_for('train_model'))
            
            # Load models and scaler
            model1 = joblib.load(training.model1_path)
            model2 = joblib.load(training.model2_path)
            scaler = joblib.load(training.scaler_path)
            
            # Prepare input
            feature_names = json.loads(training.feature_names)
            input_data = np.array([[temperature, humidity, soil_moisture, light_intensity]])
            input_scaled = scaler.transform(input_data)
            
            # Make predictions
            pred1 = model1.predict(input_scaled)[0]
            conf1 = max(model1.predict_proba(input_scaled)[0]) * 100
            
            pred2 = model2.predict(input_scaled)[0]
            conf2 = max(model2.predict_proba(input_scaled)[0]) * 100
            
            # Determine health status
            is_healthy = True if 'healthy' in str(pred1).lower() or 'healthy' in str(pred2).lower() else False
            
            # Save result
            result = TestResults(
                user_id=session['user_id'],
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                light_intensity=light_intensity,
                model1_prediction=str(pred1),
                model1_confidence=conf1,
                model2_prediction=str(pred2),
                model2_confidence=conf2,
                is_healthy=is_healthy,
                notes=notes
            )
            
            db.session.add(result)
            db.session.commit()
            
            flash('Test results saved successfully!', 'success')
            return redirect(url_for('test_results', result_id=result.id))
        
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            flash(f'Test failed: {str(e)}', 'danger')
    
    user = User.query.get(session.get('user_id'))
    return render_template('test_sensors.html', user=user)


@app.route('/test-results/<int:result_id>')
@login_required
def test_results(result_id):
    """View sensor test results"""
    result = TestResults.query.get_or_404(result_id)
    
    if result.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('test_results.html', result=result)


@app.route('/test-image', methods=['GET', 'POST'])
@login_required
def test_image():
    """Test leaf disease detection"""
    if request.method == 'POST':
        try:
            if 'image_file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['image_file']
            notes = request.form.get('notes', '')

            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not allowed_image_file(file.filename):
                return jsonify({'error': 'Unsupported file type. Please upload a PNG, JPG, JPEG or WEBP image.'}), 400

            if not app.config.get('GROQ_API_KEY'):
                return jsonify({'error': 'Server is not configured with a GROQ_API_KEY. Please contact the administrator.'}), 500

            # Save file with a unique name to avoid collisions
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename = secure_filename(f"{timestamp}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Downscale/compress large images before sending to the AI model.
            # This keeps the upload payload small so slower connections don't
            # time out, without noticeably hurting detection quality.
            try:
                with PILImage.open(filepath) as img:
                    img = img.convert('RGB')
                    max_dimension = 1280
                    if max(img.size) > max_dimension:
                        img.thumbnail((max_dimension, max_dimension), PILImage.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85, optimize=True)
                    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as img_err:
                logger.warning(f"Image resize failed, falling back to original file: {img_err}")
                with open(filepath, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

            # Analyze image
            detector = LeafDiseaseDetector(app.config['GROQ_API_KEY'])
            analysis = detector.analyze_leaf_image_base64(image_data)

            # Save result
            image_result = ImageAnalysis(
                user_id=session['user_id'],
                filename=filename,
                disease_detected=analysis['disease_detected'],
                disease_name=analysis.get('disease_name'),
                disease_type=analysis.get('disease_type'),
                severity=analysis.get('severity'),
                confidence=analysis.get('confidence'),
                symptoms=json.dumps(analysis.get('symptoms', [])),
                possible_causes=json.dumps(analysis.get('possible_causes', [])),
                treatment=json.dumps(analysis.get('treatment', [])),
                notes=notes
            )

            db.session.add(image_result)
            db.session.commit()

            return jsonify({
                'success': True,
                'redirect': url_for('image_results', image_id=image_result.id)
            })

        except Exception as e:
            error_text = str(e)
            logger.error(f"Image analysis error: {error_text}")
            if 'timeout' in error_text.lower() or 'timed out' in error_text.lower():
                return jsonify({'error': 'The AI took too long to respond. This is usually a slow/unstable internet connection. Please check your connection and try again.'}), 504
            return jsonify({'error': f'Analysis failed: {error_text}'}), 500

    user = User.query.get(session.get('user_id'))
    return render_template('test_image.html', user=user)


@app.route('/disease-library')
@login_required
def disease_library():
    """Reference library of common crop diseases, causes and prevention"""
    user = User.query.get(session.get('user_id'))
    return render_template('disease_library.html', user=user, diseases=DISEASE_LIBRARY)


@app.route('/download-sample-csv')
@login_required
def download_sample_csv():
    """Provide a sample CSV template for model training"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['temperature', 'humidity', 'soil_moisture', 'light_intensity', 'health_status'])
    writer.writerow([24.5, 65.0, 48.0, 2600, 'healthy'])
    writer.writerow([28.9, 82.0, 30.0, 1500, 'diseased'])
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_training_data.csv'}
    )


@app.route('/image-results/<int:image_id>')
@login_required
def image_results(image_id):
    """View image analysis results"""
    result = ImageAnalysis.query.get_or_404(image_id)
    
    if result.user_id != session['user_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('image_results.html', result=result)


@app.route('/results')
@login_required
def results():
    """View all results and analytics"""
    user = User.query.get(session['user_id'])
    test_results = TestResults.query.filter_by(user_id=user.id).order_by(
        TestResults.created_at.desc()).all()
    image_results = ImageAnalysis.query.filter_by(user_id=user.id).order_by(
        ImageAnalysis.created_at.desc()).all()
    
    return render_template('results.html', test_results=test_results, image_results=image_results)


@app.route('/test-log')
@login_required
def test_log():
    """View test log"""
    user = User.query.get(session['user_id'])
    tests = TestResults.query.filter_by(user_id=user.id).order_by(
        TestResults.created_at.desc()).all()
    images = ImageAnalysis.query.filter_by(user_id=user.id).order_by(
        ImageAnalysis.created_at.desc()).all()
    
    return render_template('test_log.html', tests=tests, images=images)


# ============================================================================
# API ENDPOINTS FOR DATA
# ============================================================================

@app.route('/api/charts-data')
@login_required
def charts_data():
    """API endpoint to get data for charts"""
    user_id = session['user_id']
    
    # Get test results
    tests = TestResults.query.filter_by(user_id=user_id).all()
    
    # Prepare chart data
    dates = [test.created_at.strftime('%Y-%m-%d') for test in tests]
    temps = [test.temperature for test in tests]
    humidity = [test.humidity for test in tests]
    soil_moisture = [test.soil_moisture for test in tests]
    light_intensity = [test.light_intensity for test in tests]
    
    return jsonify({
        'dates': dates,
        'temperature': temps,
        'humidity': humidity,
        'soil_moisture': soil_moisture,
        'light_intensity': light_intensity
    })


@app.route('/api/disease-stats')
@login_required
def disease_stats():
    """API endpoint for disease-type breakdown used in dashboard charts"""
    user_id = session['user_id']
    images = ImageAnalysis.query.filter_by(user_id=user_id).all()

    healthy_count = sum(1 for img in images if not img.disease_detected)
    diseased_count = sum(1 for img in images if img.disease_detected)

    type_counts: Dict[str, int] = {}
    for img in images:
        if img.disease_detected and img.disease_type:
            type_counts[img.disease_type] = type_counts.get(img.disease_type, 0) + 1

    return jsonify({
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'type_labels': list(type_counts.keys()),
        'type_values': list(type_counts.values())
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(error)}")
    return render_template('500.html'), 500


@app.errorhandler(413)
def file_too_large(error):
    """Handle file-too-large errors"""
    flash('File is too large. Maximum upload size is 16MB.', 'danger')
    return redirect(request.referrer or url_for('dashboard')), 413


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Initialize the database"""
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

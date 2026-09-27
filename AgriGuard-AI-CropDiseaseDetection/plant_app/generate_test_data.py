"""
Test Data Generator for Smart Plant Health Monitoring System
Generates sample sensor data and image analysis data for testing
"""

import csv
import random
import json
from datetime import datetime, timedelta

def generate_sensor_data(filename='sample_data.csv', rows=100):
    """Generate sample sensor data CSV file"""
    
    print(f"Generating {rows} rows of sensor data...")
    
    data = []
    for i in range(rows):
        # Generate realistic sensor readings
        temperature = random.uniform(15, 30)
        humidity = random.uniform(40, 90)
        soil_moisture = random.uniform(20, 70)
        light_intensity = random.uniform(800, 4000)
        
        # Determine health status based on conditions
        # Optimal ranges: temp 20-25, humidity 60-80, soil 40-60, light 2000-3500
        conditions_met = 0
        if 20 <= temperature <= 25:
            conditions_met += 1
        if 60 <= humidity <= 80:
            conditions_met += 1
        if 40 <= soil_moisture <= 60:
            conditions_met += 1
        if 2000 <= light_intensity <= 3500:
            conditions_met += 1
        
        # If 3+ conditions are met, plant is healthy
        health_status = 'healthy' if conditions_met >= 3 else 'diseased'
        
        data.append({
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'soil_moisture': round(soil_moisture, 1),
            'light_intensity': round(light_intensity, 1),
            'health_status': health_status
        })
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['temperature', 'humidity', 'soil_moisture', 'light_intensity', 'health_status'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Generated {filename} with {rows} rows")
    
    # Print statistics
    healthy = sum(1 for d in data if d['health_status'] == 'healthy')
    diseased = rows - healthy
    print(f"  - Healthy samples: {healthy} ({healthy/rows*100:.1f}%)")
    print(f"  - Diseased samples: {diseased} ({diseased/rows*100:.1f}%)")


def generate_test_records(count=30):
    """Generate sample test records for database"""
    
    print(f"\nGenerating {count} sample test records...")
    
    records = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(count):
        timestamp = base_date + timedelta(days=i//4)
        
        record = {
            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': round(random.uniform(18, 28), 1),
            'humidity': round(random.uniform(45, 85), 1),
            'soil_moisture': round(random.uniform(25, 65), 1),
            'light_intensity': round(random.uniform(1000, 4000), 1),
            'is_healthy': random.choice([True, False])
        }
        records.append(record)
    
    return records


def generate_image_analysis_samples(count=10):
    """Generate sample image analysis results"""
    
    print(f"\nGenerating {count} sample image analysis records...")
    
    diseases = [
        {
            'name': 'Powdery Mildew',
            'type': 'fungal',
            'severity': 'moderate',
            'symptoms': ['White powder on leaves', 'Leaf curling', 'Stunted growth'],
            'causes': ['High humidity', 'Poor air circulation', 'Warm temperatures'],
            'treatment': ['Apply fungicide', 'Improve ventilation', 'Reduce humidity']
        },
        {
            'name': 'Leaf Rust',
            'type': 'fungal',
            'severity': 'severe',
            'symptoms': ['Orange-brown spots', 'Yellowing', 'Leaf drop'],
            'causes': ['Wet foliage', 'High humidity', 'Poor drainage'],
            'treatment': ['Remove infected leaves', 'Apply rust fungicide', 'Improve drainage']
        },
        {
            'name': 'Spider Mites',
            'type': 'pest',
            'severity': 'mild',
            'symptoms': ['Fine webbing', 'Tiny dots on leaves', 'Yellowing'],
            'causes': ['Dry air', 'Warm temperatures', 'Dust accumulation'],
            'treatment': ['Increase humidity', 'Spray with water', 'Use insecticidal soap']
        },
        {
            'name': 'Nutrient Deficiency',
            'type': 'nutrient deficiency',
            'severity': 'moderate',
            'symptoms': ['Yellowing leaves', 'Poor growth', 'Pale color'],
            'causes': ['Poor soil', 'Over-watering', 'pH imbalance'],
            'treatment': ['Apply balanced fertilizer', 'Check pH', 'Adjust watering']
        }
    ]
    
    healthy = {
        'name': None,
        'type': 'healthy',
        'severity': 'none',
        'symptoms': ['No visible disease', 'Vibrant green color', 'Normal growth'],
        'causes': [],
        'treatment': ['Continue regular care', 'Monitor weekly']
    }
    
    records = []
    base_date = datetime.now() - timedelta(days=20)
    
    for i in range(count):
        timestamp = base_date + timedelta(days=i//2)
        
        # 70% healthy, 30% diseased
        if random.random() < 0.7:
            sample = healthy.copy()
            disease_detected = False
        else:
            sample = random.choice(diseases).copy()
            disease_detected = True
        
        record = {
            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'filename': f'leaf_{i+1}.jpg',
            'disease_detected': disease_detected,
            'disease_name': sample['name'],
            'disease_type': sample['type'],
            'severity': sample['severity'],
            'confidence': round(random.uniform(85, 99), 1) if disease_detected else round(random.uniform(90, 99), 1),
            'symptoms': sample['symptoms'],
            'causes': sample['causes'],
            'treatment': sample['treatment']
        }
        records.append(record)
    
    return records


def print_sample_output():
    """Print sample data for reference"""
    
    print("\n" + "="*60)
    print("SAMPLE SENSOR DATA")
    print("="*60)
    
    sensor_data = [
        {'temperature': 25.5, 'humidity': 65.3, 'soil_moisture': 45.2, 'light_intensity': 2500, 'health_status': 'healthy'},
        {'temperature': 22.3, 'humidity': 72.1, 'soil_moisture': 38.5, 'light_intensity': 1800, 'health_status': 'diseased'},
    ]
    
    for data in sensor_data:
        print(f"Temperature: {data['temperature']}°C | Humidity: {data['humidity']}% | "
              f"Soil: {data['soil_moisture']}% | Light: {data['light_intensity']} lux | "
              f"Status: {data['health_status']}")
    
    print("\n" + "="*60)
    print("SAMPLE IMAGE ANALYSIS")
    print("="*60)
    
    images = generate_image_analysis_samples(3)
    for img in images:
        print(f"\nFile: {img['filename']}")
        print(f"Disease: {img['disease_name'] or 'Healthy Leaf'}")
        print(f"Type: {img['disease_type']} | Severity: {img['severity']} | Confidence: {img['confidence']}%")
        if img['symptoms']:
            print(f"Symptoms: {', '.join(img['symptoms'][:2])}")
        if img['treatment']:
            print(f"Treatment: {', '.join(img['treatment'][:2])}")


def main():
    """Main execution"""
    
    print("\n" + "="*60)
    print("Smart Plant Health Monitoring - Test Data Generator")
    print("="*60)
    
    # Generate sensor data
    generate_sensor_data('sample_data.csv', 100)
    generate_sensor_data('sample_data_large.csv', 500)
    
    # Generate test records
    test_records = generate_test_records(30)
    print(f"✓ Generated {len(test_records)} test records")
    
    # Generate image samples
    image_records = generate_image_analysis_samples(10)
    print(f"✓ Generated {len(image_records)} image analysis records")
    
    # Print samples
    print_sample_output()
    
    print("\n" + "="*60)
    print("✓ Test data generation complete!")
    print("="*60)
    print("\nYou can now:")
    print("1. Use sample_data.csv to train models")
    print("2. Use sample_data_large.csv for larger datasets")
    print("3. Run app.py and test with the generated data")
    print("\n")


if __name__ == '__main__':
    main()

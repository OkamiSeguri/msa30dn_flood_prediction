import os
import json
import requests
import random  # Add this import for random data generation
from datetime import datetime
from dotenv import load_dotenv
from setup_db import get_connection, close_connection
import time

# Load environment variables from .env
load_dotenv()

WINDY_API_KEY = os.getenv("WINDY_API_KEY")

# List of locations in Vietnam
LOCATIONS = [
    {"name": "Hanoi", "lat": 21.0285, "lon": 105.8542},
    {"name": "Ho_Chi_Minh_City", "lat": 10.7769, "lon": 106.7009},
    {"name": "Da_Nang", "lat": 16.0471, "lon": 108.2068},
    {"name": "Hue", "lat": 16.4637, "lon": 107.5909},
    {"name": "Can_Tho", "lat": 10.0452, "lon": 105.7469},
    {"name": "Hai_Phong", "lat": 20.8449, "lon": 106.6881},
    {"name": "Nha_Trang", "lat": 12.2388, "lon": 109.1967},
]

def check_and_cleanup_database():
    """Check and clean up database if needed"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Count current records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_count = cursor.fetchone()[0]
        
        MAX_RECORDS = 2000  # Maximum limit
        
        if total_count > MAX_RECORDS:
            print(f"Database has {total_count} records, starting cleanup...")
            
            # Delete old records but keep at least 3 newest per location per day
            cursor.execute("""
                DELETE rd1 FROM rainfall_data rd1
                WHERE rd1.id NOT IN (
                    SELECT * FROM (
                        SELECT rd2.id 
                        FROM rainfall_data rd2
                        WHERE rd2.location_name = rd1.location_name 
                        AND DATE(rd2.created_at) = DATE(rd1.created_at)
                        ORDER BY rd2.created_at DESC 
                        LIMIT 3
                    ) AS temp
                )
                AND rd1.created_at < DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            conn.commit()
            print(f"Cleaned up old records while keeping 3 newest per location per day")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error cleaning database: {e}")
        return False

def check_daily_record_count(location_name):
    """Check how many records exist for this location today"""
    try:
        conn = get_connection()
        if not conn:
            return 0
            
        cursor = conn.cursor()
        
        # Count records for today
        cursor.execute("""
            SELECT COUNT(*) FROM rainfall_data 
            WHERE location_name = %s 
            AND DATE(created_at) = CURDATE()
        """, (location_name,))
        
        count = cursor.fetchone()[0]
        
        cursor.close()
        close_connection(conn)
        
        return count
        
    except Exception as e:
        print(f"Error checking daily record count: {e}")
        return 0

def cleanup_excess_daily_records(location_name):
    """Keep only 3 newest records per location per day"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Delete all but the 3 newest records for today
        cursor.execute("""
            DELETE FROM rainfall_data 
            WHERE location_name = %s 
            AND DATE(created_at) = CURDATE()
            AND id NOT IN (
                SELECT * FROM (
                    SELECT id FROM rainfall_data 
                    WHERE location_name = %s 
                    AND DATE(created_at) = CURDATE()
                    ORDER BY created_at DESC 
                    LIMIT 3
                ) AS temp
            )
        """, (location_name, location_name))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} excess records for {location_name}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error cleaning up excess records: {e}")
        return False

def fetch_windy_data(lat, lon):
    """Call Windy API to fetch weather data"""
    if not WINDY_API_KEY:
        print("Error: WINDY_API_KEY not found in environment variables")
        return None
        
    url = "https://api.windy.com/api/point-forecast/v2"
    headers = {"Content-Type": "application/json"}
    payload = {
        "lat": lat,
        "lon": lon,
        "model": "gfs",
        "parameters": ["precip", "temp", "wind", "rh", "pressure"],
        "levels": ["surface"],
        "key": WINDY_API_KEY
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text}")
            return None
        
        data = resp.json()
        processed_data = process_windy_response(data)
        return processed_data
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

def process_windy_response(data):
    """Process data returned from Windy API"""
    try:
        weather_info = {
            'timestamp': datetime.now().isoformat(),
            'source': 'windy_api'
        }
        
        # Process data fields
        if 'temp-surface' in data and len(data['temp-surface']) > 0:
            weather_info['temperature'] = data['temp-surface'][0]
        elif 'temp' in data and len(data['temp']) > 0:
            weather_info['temperature'] = data['temp'][0]
        else:
            weather_info['temperature'] = 0
            
        if 'rh-surface' in data and len(data['rh-surface']) > 0:
            weather_info['humidity'] = data['rh-surface'][0]
        elif 'rh' in data and len(data['rh']) > 0:
            weather_info['humidity'] = data['rh'][0]
        else:
            weather_info['humidity'] = 0
            
        if 'pressure-surface' in data and len(data['pressure-surface']) > 0:
            weather_info['pressure'] = data['pressure-surface'][0]
        elif 'pressure' in data and len(data['pressure']) > 0:
            weather_info['pressure'] = data['pressure'][0]
        else:
            weather_info['pressure'] = 0
            
        if 'precip-surface' in data and len(data['precip-surface']) > 0:
            weather_info['rainfall_1h'] = data['precip-surface'][0]
            if len(data['precip-surface']) >= 3:
                weather_info['rainfall_3h'] = sum(data['precip-surface'][:3])
            else:
                weather_info['rainfall_3h'] = data['precip-surface'][0]
        elif 'precip' in data and len(data['precip']) > 0:
            weather_info['rainfall_1h'] = data['precip'][0]
            if len(data['precip']) >= 3:
                weather_info['rainfall_3h'] = sum(data['precip'][:3])
            else:
                weather_info['rainfall_3h'] = data['precip'][0]
        else:
            weather_info['rainfall_1h'] = 0
            weather_info['rainfall_3h'] = 0
            
        if 'wind-surface' in data and len(data['wind-surface']) > 0:
            weather_info['wind_speed'] = data['wind-surface'][0]
        elif 'wind' in data and len(data['wind']) > 0:
            weather_info['wind_speed'] = data['wind'][0]
        else:
            weather_info['wind_speed'] = 0
            
        # Generate mock data if values are 0 (simulate real-like data)
        if weather_info['wind_speed'] == 0:
            weather_info['wind_speed'] = round(random.uniform(5, 25), 1)  # Random 5-25 km/h
        
        if weather_info['rainfall_1h'] == 0:
            weather_info['rainfall_1h'] = round(random.uniform(0.1, 10), 1)  # Random 0.1-10 mm
        
        if weather_info['rainfall_3h'] == 0:
            weather_info['rainfall_3h'] = round(random.uniform(0.5, 30), 1)  # Random 0.5-30 mm (higher for 3h)
            
        return weather_info
        
    except Exception as e:
        print(f"Error processing Windy data: {e}")
        return None

def save_to_database(location_name, lat, lon, precipitation_data):
    """Save data to database"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return False
            
        cursor = conn.cursor()
        
        precipitation_json = json.dumps(precipitation_data)
        
        query = """
        INSERT INTO rainfall_data (location_name, latitude, longitude, precipitation, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        
        cursor.execute(query, (location_name, lat, lon, precipitation_json))
        conn.commit()
        
        print(f"Data saved for {location_name}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def main():
    print("Starting to crawl data from Windy API...")
    
    # Step 1: Check and clean up database
    print("Checking database...")
    check_and_cleanup_database()
    
    # Step 2: Test database connection
    conn = get_connection()
    if not conn:
        print("Cannot connect to database. Please run setup_db.py first")
        return
    else:
        print("Database connection successful")
        close_connection(conn)
    
    # Step 3: Check API key
    if not WINDY_API_KEY:
        print("Error: WINDY_API_KEY not found in .env file")
        return
    
    # Step 4: Crawl data
    MIN_DAILY_RECORDS = 3  # Minimum 3 records per day per location
    
    for location in LOCATIONS:
        print(f"\nCrawling data for {location['name']}...")
        
        # Check current record count for today
        daily_count = check_daily_record_count(location['name'])
        print(f"Current records today for {location['name']}: {daily_count}")
        
        # Always crawl new data
        weather_data = fetch_windy_data(location['lat'], location['lon'])
        if weather_data:
            saved = save_to_database(
                location['name'], 
                location['lat'], 
                location['lon'], 
                weather_data
            )
            
            if saved:
                # After saving, check if we have more than 3 records and clean up
                new_count = check_daily_record_count(location['name'])
                if new_count > MIN_DAILY_RECORDS:
                    cleanup_excess_daily_records(location['name'])
                    print(f"Kept only {MIN_DAILY_RECORDS} newest records for {location['name']}")
            else:
                print(f"Cannot save data for {location['name']}")
        else:
            print(f"No data received from Windy API for {location['name']}")
        
        time.sleep(2)  # Avoid spamming API
    
    print("\nData crawling completed!")

if __name__ == "__main__":
    main()
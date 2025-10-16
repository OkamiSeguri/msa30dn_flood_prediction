import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from setup_db import get_connection, close_connection
import time

# Load bien moi truong tu .env
load_dotenv()

WINDY_API_KEY = os.getenv("WINDY_API_KEY")

# Danh sach cac dia diem o Viet Nam
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
    """Kiem tra va don dep database neu can"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Dem so records hien tai
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_count = cursor.fetchone()[0]
        
        MAX_RECORDS = 2000  # Gioi han toi da
        
        if total_count > MAX_RECORDS:
            print(f"Database co {total_count} records, bat dau don dep...")
            
            # Xoa 500 records cu nhat
            records_to_delete = total_count - MAX_RECORDS + 500
            cursor.execute("""
                DELETE FROM rainfall_data 
                ORDER BY created_at ASC 
                LIMIT %s
            """, (records_to_delete,))
            
            conn.commit()
            print(f"Da xoa {records_to_delete} records cu")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi don dep database: {e}")
        return False

def check_duplicate_today(location_name):
    """Kiem tra da crawl du lieu cho location hom nay chua"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Kiem tra co record nao hom nay chua
        cursor.execute("""
            SELECT COUNT(*) FROM rainfall_data 
            WHERE location_name = %s 
            AND DATE(created_at) = CURDATE()
        """, (location_name,))
        
        count = cursor.fetchone()[0]
        
        cursor.close()
        close_connection(conn)
        
        return count > 0  # True neu da co du lieu hom nay
        
    except Exception as e:
        print(f"Loi khi kiem tra duplicate: {e}")
        return False

def fetch_windy_data(lat, lon):
    """Goi Windy API de lay du lieu thoi tiet"""
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
    """Xu ly du lieu tra ve tu Windy API"""
    try:
        weather_info = {
            'timestamp': datetime.now().isoformat(),
            'source': 'windy_api'
        }
        
        # Xu ly cac truong du lieu
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
            
        return weather_info
        
    except Exception as e:
        print(f"Error processing Windy data: {e}")
        return None

def save_to_database(location_name, lat, lon, precipitation_data):
    """Luu du lieu vao database"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return False
            
        cursor = conn.cursor()
        
        precipitation_json = json.dumps(precipitation_data)
        
        query = """
        INSERT INTO rainfall_data (location_name, latitude, longitude, precipitation, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        
        cursor.execute(query, (location_name, lat, lon, precipitation_json))
        conn.commit()
        
        print(f"Da luu data cho {location_name}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi luu vao database: {e}")
        return False

def main():
    print("Bat dau crawl du lieu tu Windy API...")
    
    # Buoc 1: Kiem tra va don dep database
    print("Kiem tra database...")
    check_and_cleanup_database()
    
    # Buoc 2: Test ket noi database
    conn = get_connection()
    if not conn:
        print("Khong the ket noi database. Vui long chay setup_db.py truoc")
        return
    else:
        print("Ket noi database thanh cong")
        close_connection(conn)
    
    # Buoc 3: Kiem tra API key
    if not WINDY_API_KEY:
        print("Loi: Khong tim thay WINDY_API_KEY trong file .env")
        return
    
    # Buoc 4: Crawl du lieu
    for location in LOCATIONS:
        print(f"\nDang crawl data cho {location['name']}...")
        
        # Kiem tra da crawl hom nay chua
        if check_duplicate_today(location['name']):
            print(f"Da co du lieu cho {location['name']} hom nay, bo qua...")
            continue
        
        weather_data = fetch_windy_data(location['lat'], location['lon'])
        if weather_data:
            saved = save_to_database(
                location['name'], 
                location['lat'], 
                location['lon'], 
                weather_data
            )
            
            if not saved:
                print(f"Khong the luu data cho {location['name']}")
        else:
            print(f"Khong nhan duoc data tu Windy API cho {location['name']}")
        
        time.sleep(2)  # Tranh spam API
    
    print("\nHoan thanh crawl data!")

if __name__ == "__main__":
    main()

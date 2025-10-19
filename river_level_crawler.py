import json
import numpy as np
from datetime import datetime, timedelta
import random
import math
from setup_db import get_connection, close_connection
import time

# Updated RIVER_STATIONS with more realistic levels and lower volatility
RIVER_STATIONS = [
    {
        "location_name": "Hanoi",
        "river_name": "Red River",
        "latitude": 21.0285,
        "longitude": 105.8542,
        "normal_level": 250,  # Adjusted to realistic level
        "alert_level_1": 350,
        "alert_level_2": 450,
        "alert_level_3": 550,
        "base_flow_rate": 1200,
        "seasonal_factor": 1.2,
        "tidal_effect": False,
        "dam_controlled": True,
        "volatility": 0.08  # Reduced volatility
    },
    {
        "location_name": "Ho_Chi_Minh_City",
        "river_name": "Saigon River",
        "latitude": 10.7769,
        "longitude": 106.7009,
        "normal_level": 120,
        "alert_level_1": 180,
        "alert_level_2": 220,
        "alert_level_3": 270,
        "base_flow_rate": 800,
        "seasonal_factor": 1.4,
        "tidal_effect": True,
        "dam_controlled": False,
        "volatility": 0.12  # Reduced
    },
    {
        "location_name": "Da_Nang",
        "river_name": "Han River",
        "latitude": 16.0471,
        "longitude": 108.2068,
        "normal_level": 150,
        "alert_level_1": 200,
        "alert_level_2": 250,
        "alert_level_3": 300,
        "base_flow_rate": 600,
        "seasonal_factor": 1.3,
        "tidal_effect": True,
        "dam_controlled": True,
        "volatility": 0.10  # Reduced
    },
    {
        "location_name": "Hue",
        "river_name": "Perfume River",
        "latitude": 16.4637,
        "longitude": 107.5909,
        "normal_level": 100,
        "alert_level_1": 150,
        "alert_level_2": 200,
        "alert_level_3": 250,
        "base_flow_rate": 400,
        "seasonal_factor": 1.1,
        "tidal_effect": False,
        "dam_controlled": False,
        "volatility": 0.09  # Reduced
    },
    {
        "location_name": "Can_Tho",
        "river_name": "Mekong River",
        "latitude": 10.0452,
        "longitude": 105.7469,
        "normal_level": 200,
        "alert_level_1": 280,
        "alert_level_2": 350,
        "alert_level_3": 420,
        "base_flow_rate": 2000,
        "seasonal_factor": 1.5,
        "tidal_effect": True,
        "dam_controlled": False,
        "volatility": 0.15  # Slightly reduced
    },
    {
        "location_name": "Hai_Phong",
        "river_name": "Thai Binh River",
        "latitude": 20.8449,
        "longitude": 106.6881,
        "normal_level": 250,
        "alert_level_1": 350,
        "alert_level_2": 450,
        "alert_level_3": 550,
        "base_flow_rate": 900,
        "seasonal_factor": 1.1,
        "tidal_effect": True,
        "dam_controlled": True,
        "volatility": 0.11  # Reduced
    },
    {
        "location_name": "Nha_Trang",
        "river_name": "Cai River",
        "latitude": 12.2388,
        "longitude": 109.1967,
        "normal_level": 80,
        "alert_level_1": 120,
        "alert_level_2": 160,
        "alert_level_3": 200,
        "base_flow_rate": 300,
        "seasonal_factor": 1.2,
        "tidal_effect": False,
        "dam_controlled": False,
        "volatility": 0.14  # Reduced
    }
]

def check_and_cleanup_database():
    """Check and clean up database if needed"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Count current records
        cursor.execute("SELECT COUNT(*) FROM river_level_data")
        total_count = cursor.fetchone()[0]
        
        MAX_RECORDS = 2000  # Maximum limit
        
        if total_count > MAX_RECORDS:
            print(f"Database has {total_count} records, starting cleanup...")
            
            # Delete old records but keep at least 3 newest per location per day
            cursor.execute("""
                DELETE rd1 FROM river_level_data rd1
                WHERE rd1.id NOT IN (
                    SELECT * FROM (
                        SELECT rd2.id 
                        FROM river_level_data rd2
                        WHERE rd2.location_name = rd1.location_name 
                        AND rd2.river_name = rd1.river_name
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

def check_daily_record_count(location_name, river_name):
    """Check how many records exist for this location and river today"""
    try:
        conn = get_connection()
        if not conn:
            return 0
            
        cursor = conn.cursor()
        
        # Count records for today
        cursor.execute("""
            SELECT COUNT(*) FROM river_level_data 
            WHERE location_name = %s AND river_name = %s
            AND DATE(created_at) = CURDATE()
        """, (location_name, river_name))
        
        count = cursor.fetchone()[0]
        
        cursor.close()
        close_connection(conn)
        
        return count
        
    except Exception as e:
        print(f"Error checking daily record count: {e}")
        return 0

def cleanup_excess_daily_records(location_name, river_name):
    """Keep only 3 newest records per location per day"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Delete all but the 3 newest records for today
        cursor.execute("""
            DELETE FROM river_level_data 
            WHERE location_name = %s AND river_name = %s
            AND DATE(created_at) = CURDATE()
            AND id NOT IN (
                SELECT * FROM (
                    SELECT id FROM river_level_data 
                    WHERE location_name = %s AND river_name = %s
                    AND DATE(created_at) = CURDATE()
                    ORDER BY created_at DESC 
                    LIMIT 3
                ) AS temp
            )
        """, (location_name, river_name, location_name, river_name))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        if deleted_count > 0:
            print(f"Cleaned up {deleted_count} excess records for {location_name} - {river_name}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error cleaning up excess records: {e}")
        return False

def get_seasonal_factor():
    """Calculate seasonal factor (rainy/dry season)"""
    current_month = datetime.now().month
    
    # Rainy season in Vietnam: May-October
    if 5 <= current_month <= 10:
        # Rainy season: higher water level
        base_factor = 1.3
        # Highest in July-September
        if 7 <= current_month <= 9:
            base_factor = 1.5
    else:
        # Dry season: lower water level
        base_factor = 0.8
        # Lowest in February-April
        if 2 <= current_month <= 4:
            base_factor = 0.6
    
    # Add random variation
    random_factor = np.random.uniform(0.9, 1.1)
    return base_factor * random_factor

def get_daily_cycle_factor():
    """Calculate factor based on daily cycle"""
    current_hour = datetime.now().hour
    
    # Water level typically peaks in early morning (6-8 AM) and evening (6-8 PM)
    # Lowest at noon (12-2 PM) and late night (2-4 AM)
    
    hour_factors = {
        0: 0.95, 1: 0.90, 2: 0.85, 3: 0.88, 4: 0.92, 5: 0.98,
        6: 1.05, 7: 1.08, 8: 1.03, 9: 1.00, 10: 0.98, 11: 0.95,
        12: 0.90, 13: 0.88, 14: 0.92, 15: 0.95, 16: 0.98, 17: 1.02,
        18: 1.06, 19: 1.04, 20: 1.00, 21: 0.98, 22: 0.96, 23: 0.94
    }
    
    base_factor = hour_factors.get(current_hour, 1.0)
    
    # Add small random variation
    random_variation = np.random.uniform(0.98, 1.02)
    return base_factor * random_variation

def get_tidal_effect(station):
    """Calculate tidal effect (for rivers near the sea)"""
    if not station['tidal_effect']:
        return 1.0
    
    # Simulate tidal cycle (approximately 12.5 hours)
    current_time = datetime.now()
    hours_from_midnight = current_time.hour + current_time.minute / 60.0
    
    # Use sine function to simulate tides
    tidal_cycle = math.sin(2 * math.pi * hours_from_midnight / 12.5)
    
    # Tidal amplitude (5-15cm depending on location)
    tidal_amplitude = np.random.uniform(5, 15)
    
    # Add random tidal factor
    random_tidal = np.random.uniform(0.8, 1.2)
    
    tidal_effect = 1.0 + (tidal_cycle * tidal_amplitude * random_tidal) / station['normal_level']
    
    return max(0.8, min(1.2, tidal_effect))

def get_weather_impact_advanced(weather_data, station):
    """Calculate detailed weather impact with reduced impact"""
    if not weather_data:
        season_factor = get_seasonal_factor()
        
        if season_factor > 1.2:  # Rainy season
            rainfall_1h = np.random.exponential(8) * np.random.uniform(0.5, 2.0)
            rainfall_3h = rainfall_1h * np.random.uniform(2, 4)
            humidity = np.random.uniform(75, 95)
            pressure = np.random.uniform(995, 1010)
        else:  # Dry season
            rainfall_1h = np.random.exponential(1) * np.random.uniform(0, 1.5)
            rainfall_3h = rainfall_1h * np.random.uniform(1, 2.5)
            humidity = np.random.uniform(50, 75)
            pressure = np.random.uniform(1010, 1025)
        
        wind_speed = np.random.uniform(5, 25)
    else:
        rainfall_1h = weather_data.get('rainfall_1h', 0)
        rainfall_3h = weather_data.get('rainfall_3h', 0)
        humidity = weather_data.get('humidity', 70)
        pressure = weather_data.get('pressure', 1013)
        wind_speed = weather_data.get('wind_speed', 10)
    
    # Reduced rain impact coefficients
    rain_impact = 0
    
    if rainfall_1h > 0:
        if rainfall_1h > 20:
            rain_impact += rainfall_1h * 2.0  # Reduced from 4.0
        elif rainfall_1h > 10:
            rain_impact += rainfall_1h * 1.5  # Reduced from 3.0
        elif rainfall_1h > 5:
            rain_impact += rainfall_1h * 1.2  # Reduced from 2.5
        else:
            rain_impact += rainfall_1h * 1.0  # Reduced from 2.0
    
    if rainfall_3h > rainfall_1h:
        accumulated_rain = rainfall_3h - rainfall_1h
        rain_impact += accumulated_rain * 0.7  # Reduced from 1.5
    
    # Reduced humidity and pressure impacts
    if humidity > 90:
        rain_impact *= 1.2  # Reduced from 1.4
    elif humidity > 80:
        rain_impact *= 1.1  # Reduced from 1.2
    elif humidity < 50:
        rain_impact *= 0.8  # Reduced from 0.6
    
    if pressure < 990:
        rain_impact *= 1.4  # Reduced from 1.8
    elif pressure < 1000:
        rain_impact *= 1.2  # Reduced from 1.5
    elif pressure < 1005:
        rain_impact *= 1.1  # Reduced from 1.2
    elif pressure > 1020:
        rain_impact *= 0.9  # Reduced from 0.7
    
    # Reduced wind impact
    if wind_speed > 40:
        rain_impact *= 0.95  # Reduced from 0.8
    elif wind_speed > 25:
        rain_impact *= 0.97  # Reduced from 0.9
    elif wind_speed < 5:
        rain_impact *= 1.05  # Reduced from 1.1
    
    return rain_impact

def get_human_activities_impact(station):
    """Simulate human activity impact"""
    impact = 0
    
    # Dam release (if applicable)
    if station['dam_controlled']:
        # 5% chance of dam release
        if np.random.random() < 0.05:
            dam_release = np.random.uniform(20, 60)  # Release 20-60cm
            impact += dam_release
            print(f"  [Dam Control] Water release: +{dam_release:.1f}cm")
    
    # Sand and gravel mining
    if np.random.random() < 0.02:  # 2% chance
        mining_impact = np.random.uniform(-10, -25)  # Reduces water level
        impact += mining_impact
        print(f"  [Mining] Impact: {mining_impact:.1f}cm")
    
    # Hydraulic construction
    if np.random.random() < 0.01:  # 1% chance
        construction_impact = np.random.uniform(-5, 15)
        impact += construction_impact
        if construction_impact > 0:
            print(f"  [Construction] Obstructs flow: +{construction_impact:.1f}cm")
        else:
            print(f"  [Construction] Creates drainage: {construction_impact:.1f}cm")
    
    return impact

def get_geological_factors(station):
    """Calculate geological factors"""
    impact = 0
    
    # Riverbank erosion (rare)
    if np.random.random() < 0.005:  # 0.5% chance
        erosion_impact = np.random.uniform(8, 20)
        impact += erosion_impact
        print(f"  [Warning] Riverbank erosion: +{erosion_impact:.1f}cm")
    
    # Sedimentation (gradual impact)
    sedimentation = np.random.uniform(-2, 3)  # Usually slightly increases water level
    impact += sedimentation
    
    return impact

def calculate_natural_flow_change(prev_level, station):
    """Calculate natural flow change with reduced variation"""
    prev_level = float(prev_level)
    normal_level = float(station['normal_level'])
    volatility = float(station['volatility'])
    
    level_ratio = prev_level / normal_level
    
    # Reduced drainage speeds for stability
    if level_ratio > 2.0:
        natural_decline = np.random.uniform(5, 10)  # Reduced from 15-25
    elif level_ratio > 1.5:
        natural_decline = np.random.uniform(4, 8)   # Reduced from 10-18
    elif level_ratio > 1.2:
        natural_decline = np.random.uniform(2, 6)   # Reduced from 6-12
    elif level_ratio > 0.8:
        natural_decline = np.random.uniform(1, 4)   # Reduced from 3-8
    else:
        natural_decline = np.random.uniform(0.5, 2) # Reduced from 1-4
    
    # Reduced volatility factor
    volatility_factor = 1 + np.random.uniform(-volatility/2, volatility/2)  # Halved range
    
    return float(natural_decline * volatility_factor)

def get_latest_weather_data(location_name):
    """Retrieve the latest weather data from the database"""
    try:
        conn = get_connection()
        if not conn:
            return None
            
        cursor = conn.cursor()
        
        # SỬA QUERY ĐỂ LẤY THÊM THÔNG TIN CẦN THIẾT
        query = """
        SELECT temperature, humidity, pressure, wind_speed, precipitation 
        FROM rainfall_data 
        WHERE location_name = %s 
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (location_name,))
        result = cursor.fetchone()
        
        cursor.close()
        close_connection(conn)
        
        if result:
            # CHUYỂN ĐỔI TẤT CẢ DECIMAL THÀNH FLOAT
            temperature = float(result[0]) if result[0] is not None else 26.0
            humidity = float(result[1]) if result[1] is not None else 70.0
            pressure = float(result[2]) if result[2] is not None else 1013.0
            wind_speed = float(result[3]) if result[3] is not None else 10.0
            
            # XỬ LÝ PRECIPITATION (có thể là JSON string)
            precipitation_data = result[4]
            if precipitation_data:
                try:
                    if isinstance(precipitation_data, str):
                        precip_json = json.loads(precipitation_data)
                    else:
                        precip_json = precipitation_data
                    
                    rainfall_1h = float(precip_json.get('rainfall_1h', 0)) if precip_json.get('rainfall_1h') else 0
                    rainfall_3h = float(precip_json.get('rainfall_3h', 0)) if precip_json.get('rainfall_3h') else 0
                    
                except (json.JSONDecodeError, TypeError, AttributeError):
                    rainfall_1h = 0
                    rainfall_3h = 0
            else:
                rainfall_1h = 0
                rainfall_3h = 0
            
            return {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'rainfall_1h': rainfall_1h,
                'rainfall_3h': rainfall_3h
            }
        
        return None
        
    except Exception as e:
        print(f"Error retrieving weather data: {e}")
        return None

def get_previous_river_level(location_name, river_name):
    """Retrieve the previous river level to determine trend"""
    try:
        conn = get_connection()
        if not conn:
            return None
            
        cursor = conn.cursor()
        
        query = """
        SELECT water_level, trend FROM river_level_data 
        WHERE location_name = %s AND river_name = %s
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (location_name, river_name))
        result = cursor.fetchone()
        
        cursor.close()
        close_connection(conn)
        
        if result:
            # CHUYỂN ĐỔI DECIMAL THÀNH FLOAT
            water_level = float(result[0]) if result[0] is not None else None
            trend = result[1]
            return (water_level, trend)
        
        return None
        
    except Exception as e:
        print(f"Error retrieving previous water level: {e}")
        return None

def simulate_river_level(station, weather_data):
    """Simulate river water level with smoothing for realistic changes"""
    
    # Retrieve previous water level
    prev_data = get_previous_river_level(station['location_name'], station['river_name'])
    
    if prev_data and prev_data[0] is not None:
        prev_level, prev_trend = prev_data
        prev_level = float(prev_level)
    else:
        seasonal_adj = get_seasonal_factor()
        prev_level = float(station['normal_level']) * seasonal_adj * np.random.uniform(0.9, 1.1)
        prev_trend = 'stable'
    
    print(f"  Previous water level: {prev_level:.1f}cm (trend: {prev_trend})")
    
    # Calculate all impacts (with reduced factors)
    if weather_data:
        safe_weather_data = {}
        for key, value in weather_data.items():
            if value is not None:
                try:
                    safe_weather_data[key] = float(value)
                except (ValueError, TypeError):
                    safe_weather_data[key] = 0.0
            else:
                safe_weather_data[key] = 0.0
        weather_impact = get_weather_impact_advanced(safe_weather_data, station)
    else:
        weather_impact = get_weather_impact_advanced(None, station)
    
    seasonal_factor = get_seasonal_factor()
    seasonal_impact = (seasonal_factor - 1) * float(station['normal_level']) * 0.15  # Reduced from 0.3
    
    daily_cycle = get_daily_cycle_factor()
    daily_impact = (daily_cycle - 1) * float(station['normal_level']) * 0.05  # Reduced from 0.1
    
    tidal_factor = get_tidal_effect(station)
    tidal_impact = (tidal_factor - 1) * float(station['normal_level']) * 0.5  # Reduced tidal impact
    
    human_impact = get_human_activities_impact(station)
    geological_impact = get_geological_factors(station)
    natural_decline = calculate_natural_flow_change(prev_level, station)
    
    momentum_impact = 0
    if prev_trend == 'rising':
        momentum_impact = np.random.uniform(1, 4)  # Reduced from 2-8
    elif prev_trend == 'falling':
        momentum_impact = np.random.uniform(-4, -1)  # Reduced from -8 to -2
    
    total_change = float(weather_impact + seasonal_impact + daily_impact + 
                        tidal_impact + human_impact + geological_impact + 
                        momentum_impact - natural_decline)
    
    # Apply smoothing: limit change to max 8cm per crawl
    max_change = 8.0  # Maximum change per crawl (cm)
    if abs(total_change) > max_change:
        total_change = max_change if total_change > 0 else -max_change
        print(f"  [Smoothing] Limited change to {total_change:+.1f}cm")
    
    new_level = float(prev_level) + total_change
    
    # Reduced measurement noise
    measurement_noise = np.random.uniform(-1, 1)  # Reduced from -2 to 2
    new_level += measurement_noise
    
    # Ensure reasonable limits
    min_level = float(station['normal_level']) * 0.3  # Minimum 30% of normal level
    max_level = float(station['alert_level_3']) * 1.2  # Maximum 120% of alert level 3
    
    new_level = max(min_level, min(new_level, max_level))
    
    # Determine trend with smaller threshold
    level_change = new_level - prev_level
    if level_change > 2:  # Reduced from 3
        trend = 'rising'
    elif level_change < -2:  # Reduced from -3
        trend = 'falling'
    else:
        trend = 'stable'
    
    # Calculate flow rate with reduced variation
    level_ratio = new_level / float(station['normal_level'])
    flow_rate = float(station['base_flow_rate']) * (level_ratio ** 1.5)  # Reduced exponent from 1.8
    
    if weather_data and weather_data.get('rainfall_1h', 0) > 15:
        flow_rate *= 1.15  # Reduced from 1.3
    
    flow_variation = np.random.uniform(0.92, 1.08)  # Reduced range from 0.85-1.15
    flow_rate *= flow_variation
    
    # Print impacts (optional, can be commented out for less output)
    print(f"  Total change: {total_change:+.1f}cm (limited to ±{max_change}cm)")
    
    return {
        'water_level': round(float(new_level), 2),
        'flow_rate': round(float(flow_rate), 2),
        'trend': trend,
        'weather_impact': round(float(weather_impact), 2),
        'level_change': round(float(level_change), 2),
        'seasonal_factor': round(float(seasonal_factor), 3),
        'tidal_factor': round(float(tidal_factor), 3) if station['tidal_effect'] else 1.0
    }

def save_river_level_data(station, river_data):
    """Save river water level data to the database"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return False
            
        cursor = conn.cursor()
        
        query = """
        INSERT INTO river_level_data 
        (location_name, river_name, latitude, longitude, water_level, 
         normal_level, alert_level_1, alert_level_2, alert_level_3, 
         flow_rate, trend, data_source, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        values = (
            station['location_name'],
            station['river_name'],
            station['latitude'],
            station['longitude'],
            river_data['water_level'],
            station['normal_level'],
            station['alert_level_1'],
            station['alert_level_2'],
            station['alert_level_3'],
            river_data['flow_rate'],
            river_data['trend'],
            'simulated_advanced'
        )
        
        cursor.execute(query, values)
        conn.commit();
        
        print(f"Successfully saved water level data for {station['location_name']} - {station['river_name']}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error saving water level data: {e}")
        return False

def main():
    print("=== STARTING RIVER WATER LEVEL CRAWL (ADVANCED) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check and clean up database
    print("Checking database...")
    check_and_cleanup_database()
    
    # Check database connection
    conn = get_connection()
    if not conn:
        print("Cannot connect to database. Please run setup_db.py first")
        return
    else:
        print("Database connection successful")
        close_connection(conn)
    
    success_count = 0
    total_stations = len(RIVER_STATIONS)
    MIN_DAILY_RECORDS = 3  # Minimum 3 records per day per location
    
    # Crawl data for each measurement station
    for i, station in enumerate(RIVER_STATIONS, 1):
        print(f"\n[{i}/{total_stations}] Processing {station['location_name']} - {station['river_name']}...")
        
        # Check current record count for today
        daily_count = check_daily_record_count(station['location_name'], station['river_name'])
        print(f"Current records today for {station['location_name']} - {station['river_name']}: {daily_count}")
        
        try:
            # Retrieve corresponding weather data
            weather_data = get_latest_weather_data(station['location_name'])
            
            if weather_data:
                print(f"  Using real weather data")
            else:
                print(f"  Generating simulated weather data")
            
            # Simulate river water level
            river_data = simulate_river_level(station, weather_data)
            
            # Display detailed information
            print(f"  Results:")
            print(f"    Water level: {river_data['water_level']:.1f}cm (Normal: {station['normal_level']}cm)")
            print(f"    Flow rate: {river_data['flow_rate']:.1f}m³/s")
            print(f"    Trend: {river_data['trend']}")
            print(f"    Change: {river_data['level_change']:+.1f}cm")
            
            # Check alert levels
            alert_level = 0
            if river_data['water_level'] >= station['alert_level_3']:
                print(f"  ALERT LEVEL 3: Dangerous! ({river_data['water_level']:.1f}cm >= {station['alert_level_3']}cm)")
                alert_level = 3
            elif river_data['water_level'] >= station['alert_level_2']:
                print(f"  ALERT LEVEL 2: High! ({river_data['water_level']:.1f}cm >= {station['alert_level_2']}cm)")
                alert_level = 2
            elif river_data['water_level'] >= station['alert_level_1']:
                print(f"  ALERT LEVEL 1: Attention! ({river_data['water_level']:.1f}cm >= {station['alert_level_1']}cm)")
                alert_level = 1
            else:
                print(f"  Normal")
            
            # Save to database
            saved = save_river_level_data(station, river_data)
            
            if saved:
                success_count += 1
                print(f"  Saved successfully")
                
                # After saving, check if we have more than 3 records and clean up
                new_count = check_daily_record_count(station['location_name'], station['river_name'])
                if new_count > MIN_DAILY_RECORDS:
                    cleanup_excess_daily_records(station['location_name'], station['river_name'])
                    print(f"  Kept only {MIN_DAILY_RECORDS} newest records for {station['location_name']} - {station['river_name']}")
            else:
                print(f"  Failed to save data")
            
        except Exception as e:
            print(f"  Error processing {station['location_name']}: {e}")
        
        # Random delay to simulate real-time
        delay = np.random.uniform(1, 3)
        time.sleep(delay)
    
    print(f"\n=== COMPLETED RIVER WATER LEVEL CRAWL ===")
    print(f"Success: {success_count}/{total_stations} stations")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
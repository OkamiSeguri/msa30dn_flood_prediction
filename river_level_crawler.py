import json
import numpy as np
from datetime import datetime, timedelta
import random
import math
from setup_db import get_connection, close_connection
import time

# Information about major rivers and measurement stations in Vietnam
RIVER_STATIONS = [
    {
        "location_name": "Hanoi",
        "river_name": "Red River",
        "latitude": 21.0285,
        "longitude": 105.8542,
        "normal_level": 300,  # cm
        "alert_level_1": 450,
        "alert_level_2": 550,
        "alert_level_3": 650,
        "base_flow_rate": 1200,  # m3/s
        "seasonal_factor": 1.2,  # Seasonal factor (higher during rainy season)
        "tidal_effect": False,   # Not affected by tides
        "dam_controlled": True,  # Dam-regulated
        "volatility": 0.15       # Volatility (15%)
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
        "tidal_effect": True,    # Affected by tides
        "dam_controlled": False,
        "volatility": 0.25
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
        "volatility": 0.20
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
        "volatility": 0.18
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
        "volatility": 0.30
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
        "volatility": 0.22
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
        "volatility": 0.28
    }
]

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
    """Calculate detailed weather impact"""
    if not weather_data:
        # Generate random but realistic weather data
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
    
    # Calculate complex impact
    rain_impact = 0
    
    # Instant rain impact (1h)
    if rainfall_1h > 0:
        # Heavy rain has nonlinear impact
        if rainfall_1h > 20:
            rain_impact += rainfall_1h * 4.0  # Heavy rain has strong impact
        elif rainfall_1h > 10:
            rain_impact += rainfall_1h * 3.0
        elif rainfall_1h > 5:
            rain_impact += rainfall_1h * 2.5
        else:
            rain_impact += rainfall_1h * 2.0
    
    # Accumulated rain impact (3h)
    if rainfall_3h > rainfall_1h:
        accumulated_rain = rainfall_3h - rainfall_1h
        rain_impact += accumulated_rain * 1.5
    
    # Adjust by soil moisture
    if humidity > 90:
        rain_impact *= 1.4  # Fully saturated soil
    elif humidity > 80:
        rain_impact *= 1.2  # Wet soil
    elif humidity < 50:
        rain_impact *= 0.6  # Dry soil, absorbs more water
    
    # Pressure impact (storms, tropical depressions)
    if pressure < 990:
        rain_impact *= 1.8  # Strong storm
    elif pressure < 1000:
        rain_impact *= 1.5  # Tropical depression
    elif pressure < 1005:
        rain_impact *= 1.2  # Bad weather
    elif pressure > 1020:
        rain_impact *= 0.7  # Good weather, less rain
    
    # Wind impact (evaporation and drainage)
    if wind_speed > 40:
        rain_impact *= 0.8  # Very strong wind, fast drainage
    elif wind_speed > 25:
        rain_impact *= 0.9  # Strong wind
    elif wind_speed < 5:
        rain_impact *= 1.1  # Weak wind, water stagnation
    
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
    """Calculate natural flow change"""
    # Natural drainage speed depends on current water level
    level_ratio = prev_level / station['normal_level']
    
    # Higher water level drains faster
    if level_ratio > 2.0:
        natural_decline = np.random.uniform(15, 25)  # Very fast drainage
    elif level_ratio > 1.5:
        natural_decline = np.random.uniform(10, 18)  # Fast drainage
    elif level_ratio > 1.2:
        natural_decline = np.random.uniform(6, 12)   # Normal drainage
    elif level_ratio > 0.8:
        natural_decline = np.random.uniform(3, 8)    # Slow drainage
    else:
        natural_decline = np.random.uniform(1, 4)    # Very slow drainage
    
    # Add random factor based on river characteristics
    volatility_factor = 1 + np.random.uniform(-station['volatility'], station['volatility'])
    
    return natural_decline * volatility_factor

def get_latest_weather_data(location_name):
    """Retrieve the latest weather data from the database"""
    try:
        conn = get_connection()
        if not conn:
            return None
            
        cursor = conn.cursor()
        
        query = """
        SELECT precipitation FROM rainfall_data 
        WHERE location_name = %s 
        ORDER BY created_at DESC 
        LIMIT 1
        """
        
        cursor.execute(query, (location_name,))
        result = cursor.fetchone()
        
        cursor.close()
        close_connection(conn)
        
        if result:
            return json.loads(result[0])
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
        
        return result
        
    except Exception as e:
        print(f"Error retrieving previous water level: {e}")
        return None

def simulate_river_level(station, weather_data):
    """Simulate river water level with various realistic factors"""
    
    # Retrieve previous water level
    prev_data = get_previous_river_level(station['location_name'], station['river_name'])
    
    if prev_data:
        prev_level, prev_trend = prev_data
    else:
        # If no data, start with a random level near normal
        seasonal_adj = get_seasonal_factor()
        prev_level = station['normal_level'] * seasonal_adj * np.random.uniform(0.8, 1.2)
        prev_trend = 'stable'
    
    print(f"  Previous water level: {prev_level:.1f}cm (trend: {prev_trend})")
    
    # 1. Weather factor
    weather_impact = get_weather_impact_advanced(weather_data, station)
    
    # 2. Seasonal factor
    seasonal_factor = get_seasonal_factor()
    seasonal_impact = (seasonal_factor - 1) * station['normal_level'] * 0.3
    
    # 3. Daily cycle
    daily_cycle = get_daily_cycle_factor()
    daily_impact = (daily_cycle - 1) * station['normal_level'] * 0.1
    
    # 4. Tidal effect
    tidal_factor = get_tidal_effect(station)
    tidal_impact = (tidal_factor - 1) * station['normal_level']
    
    # 5. Human activities
    human_impact = get_human_activities_impact(station)
    
    # 6. Geological factors
    geological_impact = get_geological_factors(station)
    
    # 7. Natural drainage
    natural_decline = calculate_natural_flow_change(prev_level, station)
    
    # 8. Momentum (trend continuation)
    momentum_impact = 0
    if prev_trend == 'rising':
        momentum_impact = np.random.uniform(2, 8)
    elif prev_trend == 'falling':
        momentum_impact = np.random.uniform(-8, -2)
    
    # Aggregate all factors
    total_change = (weather_impact + seasonal_impact + daily_impact + 
                   tidal_impact + human_impact + geological_impact + 
                   momentum_impact - natural_decline)
    
    # Calculate new water level
    new_level = prev_level + total_change
    
    # Add small random noise (measurement error)
    measurement_noise = np.random.uniform(-2, 2)
    new_level += measurement_noise
    
    # Ensure reasonable limits
    min_level = station['normal_level'] * 0.2  # Minimum 20% of normal level
    max_level = station['alert_level_3'] * 1.3  # Maximum 130% of alert level 3
    
    new_level = max(min_level, min(new_level, max_level))
    
    # Determine new trend
    level_change = new_level - prev_level
    if level_change > 3:
        trend = 'rising'
    elif level_change < -3:
        trend = 'falling'
    else:
        trend = 'stable'
    
    # Calculate flow rate (more complex)
    level_ratio = new_level / station['normal_level']
    
    # Flow rate increases exponentially with water level
    flow_rate = station['base_flow_rate'] * (level_ratio ** 1.8)
    
    # Adjust for weather
    if weather_data and weather_data.get('rainfall_1h', 0) > 15:
        flow_rate *= 1.3  # Heavy rain increases flow rate
    
    # Add flow rate variation
    flow_variation = np.random.uniform(0.85, 1.15)
    flow_rate *= flow_variation
    
    # Print detailed impacts
    print(f"  Impact details:")
    print(f"    - Weather: {weather_impact:+.1f}cm")
    print(f"    - Seasonal: {seasonal_impact:+.1f}cm")
    print(f"    - Daily cycle: {daily_impact:+.1f}cm")
    if station['tidal_effect']:
        print(f"    - Tidal: {tidal_impact:+.1f}cm")
    if human_impact != 0:
        print(f"    - Human: {human_impact:+.1f}cm")
    if geological_impact > 5:
        print(f"    - Geological: {geological_impact:+.1f}cm")
    print(f"    - Momentum: {momentum_impact:+.1f}cm")
    print(f"    - Natural decline: -{natural_decline:.1f}cm")
    print(f"    - Total change: {total_change:+.1f}cm")
    
    return {
        'water_level': round(new_level, 2),
        'flow_rate': round(flow_rate, 2),
        'trend': trend,
        'weather_impact': round(weather_impact, 2),
        'level_change': round(level_change, 2),
        'seasonal_factor': round(seasonal_factor, 3),
        'tidal_factor': round(tidal_factor, 3) if station['tidal_effect'] else 1.0
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
        conn.commit()
        
        print(f"Successfully saved water level data for {station['location_name']} - {station['river_name']}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error saving water level data: {e}")
        return False

def check_duplicate_recent(location_name, river_name, hours=1):
    """Check for recent data within the last few hours"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        query = """
        SELECT COUNT(*) FROM river_level_data 
        WHERE location_name = %s AND river_name = %s
        AND created_at >= DATE_SUB(NOW(), INTERVAL %s HOUR)
        """
        
        cursor.execute(query, (location_name, river_name, hours))
        count = cursor.fetchone()[0]
        
        cursor.close()
        close_connection(conn)
        
        return count > 0
        
    except Exception as e:
        print(f"Error checking duplicate: {e}")
        return False

def main():
    print("=== STARTING RIVER WATER LEVEL CRAWL (ADVANCED) ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    # Crawl data for each measurement station
    for i, station in enumerate(RIVER_STATIONS, 1):
        print(f"\n[{i}/{total_stations}] Processing {station['location_name']} - {station['river_name']}...")
        
        # Check if data was recently crawled (within 1 hour)
        if check_duplicate_recent(station['location_name'], station['river_name'], 1):
            print(f"Recent data exists for {station['location_name']}, skipping...")
            continue
        
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
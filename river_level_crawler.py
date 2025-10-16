import json
import numpy as np
from datetime import datetime, timedelta
import random
import math
from setup_db import get_connection, close_connection
import time

# Thông tin các sông chính và trạm đo ở Việt Nam
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
        "seasonal_factor": 1.2,  # Hệ số mùa (mùa mưa cao hơn)
        "tidal_effect": False,   # Không bị ảnh hưởng thủy triều
        "dam_controlled": True,  # Có đập điều tiết
        "volatility": 0.15       # Độ biến động (15%)
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
        "tidal_effect": True,    # Bị ảnh hưởng thủy triều
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
    """Tính hệ số theo mùa (mùa mưa/khô)"""
    current_month = datetime.now().month
    
    # Mùa mưa ở Việt Nam: tháng 5-10
    if 5 <= current_month <= 10:
        # Mùa mưa: mực nước cao hơn
        base_factor = 1.3
        # Tháng 7-9 là cao nhất
        if 7 <= current_month <= 9:
            base_factor = 1.5
    else:
        # Mùa khô: mực nước thấp hơn
        base_factor = 0.8
        # Tháng 2-4 là thấp nhất
        if 2 <= current_month <= 4:
            base_factor = 0.6
    
    # Thêm biến động ngẫu nhiên
    random_factor = np.random.uniform(0.9, 1.1)
    return base_factor * random_factor

def get_daily_cycle_factor():
    """Tính hệ số theo chu kỳ trong ngày"""
    current_hour = datetime.now().hour
    
    # Mực nước thường cao nhất vào sáng sớm (6-8h) và chiều tối (18-20h)
    # Thấp nhất vào trưa (12-14h) và đêm khuya (2-4h)
    
    hour_factors = {
        0: 0.95, 1: 0.90, 2: 0.85, 3: 0.88, 4: 0.92, 5: 0.98,
        6: 1.05, 7: 1.08, 8: 1.03, 9: 1.00, 10: 0.98, 11: 0.95,
        12: 0.90, 13: 0.88, 14: 0.92, 15: 0.95, 16: 0.98, 17: 1.02,
        18: 1.06, 19: 1.04, 20: 1.00, 21: 0.98, 22: 0.96, 23: 0.94
    }
    
    base_factor = hour_factors.get(current_hour, 1.0)
    
    # Thêm biến động nhỏ
    random_variation = np.random.uniform(0.98, 1.02)
    return base_factor * random_variation

def get_tidal_effect(station):
    """Tính ảnh hưởng thủy triều (cho các sông gần biển)"""
    if not station['tidal_effect']:
        return 1.0
    
    # Mô phỏng chu kỳ thủy triều (khoảng 12.5 giờ)
    current_time = datetime.now()
    hours_from_midnight = current_time.hour + current_time.minute / 60.0
    
    # Sử dụng hàm sin để mô phỏng thủy triều
    tidal_cycle = math.sin(2 * math.pi * hours_from_midnight / 12.5)
    
    # Biên độ thủy triều (5-15cm tùy vị trí)
    tidal_amplitude = np.random.uniform(5, 15)
    
    # Thêm yếu tố ngẫu nhiên cho thủy triều
    random_tidal = np.random.uniform(0.8, 1.2)
    
    tidal_effect = 1.0 + (tidal_cycle * tidal_amplitude * random_tidal) / station['normal_level']
    
    return max(0.8, min(1.2, tidal_effect))

def get_weather_impact_advanced(weather_data, station):
    """Tính toán ảnh hưởng thời tiết chi tiết hơn"""
    if not weather_data:
        # Tạo dữ liệu thời tiết ngẫu nhiên nhưng thực tế
        season_factor = get_seasonal_factor()
        
        if season_factor > 1.2:  # Mùa mưa
            rainfall_1h = np.random.exponential(8) * np.random.uniform(0.5, 2.0)
            rainfall_3h = rainfall_1h * np.random.uniform(2, 4)
            humidity = np.random.uniform(75, 95)
            pressure = np.random.uniform(995, 1010)
        else:  # Mùa khô
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
    
    # Tính toán ảnh hưởng phức tạp
    rain_impact = 0
    
    # Ảnh hưởng mưa tức thời (1h)
    if rainfall_1h > 0:
        # Mưa lớn có tác động phi tuyến
        if rainfall_1h > 20:
            rain_impact += rainfall_1h * 4.0  # Mưa lớn tác động mạnh
        elif rainfall_1h > 10:
            rain_impact += rainfall_1h * 3.0
        elif rainfall_1h > 5:
            rain_impact += rainfall_1h * 2.5
        else:
            rain_impact += rainfall_1h * 2.0
    
    # Ảnh hưởng mưa tích lũy (3h)
    if rainfall_3h > rainfall_1h:
        accumulated_rain = rainfall_3h - rainfall_1h
        rain_impact += accumulated_rain * 1.5
    
    # Điều chỉnh theo độ ẩm đất
    if humidity > 90:
        rain_impact *= 1.4  # Đất bão hòa hoàn toàn
    elif humidity > 80:
        rain_impact *= 1.2  # Đất ẩm
    elif humidity < 50:
        rain_impact *= 0.6  # Đất khô, hấp thụ nhiều nước
    
    # Ảnh hưởng áp suất (bão, áp thấp nhiệt đới)
    if pressure < 990:
        rain_impact *= 1.8  # Bão mạnh
    elif pressure < 1000:
        rain_impact *= 1.5  # Áp thấp nhiệt đới
    elif pressure < 1005:
        rain_impact *= 1.2  # Thời tiết xấu
    elif pressure > 1020:
        rain_impact *= 0.7  # Thời tiết đẹp, ít mưa
    
    # Ảnh hưởng gió (bay hơi và thoát nước)
    if wind_speed > 40:
        rain_impact *= 0.8  # Gió rất mạnh, thoát nước nhanh
    elif wind_speed > 25:
        rain_impact *= 0.9  # Gió mạnh
    elif wind_speed < 5:
        rain_impact *= 1.1  # Gió yếu, nước đọng lại
    
    return rain_impact

def get_human_activities_impact(station):
    """Mô phỏng tác động của hoạt động con người"""
    impact = 0
    
    # Xả nước từ đập (nếu có)
    if station['dam_controlled']:
        # 5% cơ hội xả nước từ đập
        if np.random.random() < 0.05:
            dam_release = np.random.uniform(20, 60)  # Xả 20-60cm
            impact += dam_release
            print(f"  [Đập điều tiết] Xả nước: +{dam_release:.1f}cm")
    
    # Hoạt động khai thác cát, sỏi
    if np.random.random() < 0.02:  # 2% cơ hội
        mining_impact = np.random.uniform(-10, -25)  # Làm giảm mực nước
        impact += mining_impact
        print(f"  [Khai thác] Ảnh hưởng: {mining_impact:.1f}cm")
    
    # Xây dựng công trình thủy lợi
    if np.random.random() < 0.01:  # 1% cơ hội
        construction_impact = np.random.uniform(-5, 15)
        impact += construction_impact
        if construction_impact > 0:
            print(f"  [Xây dựng] Cản trở dòng chảy: +{construction_impact:.1f}cm")
        else:
            print(f"  [Xây dựng] Tạo kênh thoát: {construction_impact:.1f}cm")
    
    return impact

def get_geological_factors(station):
    """Tính toán yếu tố địa chất"""
    impact = 0
    
    # Sạt lở bờ sông (hiếm)
    if np.random.random() < 0.005:  # 0.5% cơ hội
        erosion_impact = np.random.uniform(8, 20)
        impact += erosion_impact
        print(f"  [Cảnh báo] Sạt lở bờ sông: +{erosion_impact:.1f}cm")
    
    # Bồi lắng (ảnh hưởng dần)
    sedimentation = np.random.uniform(-2, 3)  # Thường tăng mực nước chút ít
    impact += sedimentation
    
    return impact

def calculate_natural_flow_change(prev_level, station):
    """Tính toán sự thay đổi tự nhiên của dòng chảy"""
    # Tốc độ thoát nước tự nhiên phụ thuộc vào mực nước hiện tại
    level_ratio = prev_level / station['normal_level']
    
    # Nước cao thì thoát nhanh hơn
    if level_ratio > 2.0:
        natural_decline = np.random.uniform(15, 25)  # Thoát rất nhanh
    elif level_ratio > 1.5:
        natural_decline = np.random.uniform(10, 18)  # Thoát nhanh
    elif level_ratio > 1.2:
        natural_decline = np.random.uniform(6, 12)   # Thoát bình thường
    elif level_ratio > 0.8:
        natural_decline = np.random.uniform(3, 8)    # Thoát chậm
    else:
        natural_decline = np.random.uniform(1, 4)    # Mực thấp, thoát rất chậm
    
    # Thêm yếu tố ngẫu nhiên theo đặc tính sông
    volatility_factor = 1 + np.random.uniform(-station['volatility'], station['volatility'])
    
    return natural_decline * volatility_factor

def get_latest_weather_data(location_name):
    """Lấy dữ liệu thời tiết mới nhất từ database"""
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
        print(f"Lỗi khi lấy dữ liệu thời tiết: {e}")
        return None

def get_previous_river_level(location_name, river_name):
    """Lấy mực nước sông lần trước để tính xu hướng"""
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
        print(f"Lỗi khi lấy mực nước trước: {e}")
        return None

def simulate_river_level(station, weather_data):
    """Mô phỏng mực nước sông với nhiều yếu tố thực tế"""
    
    # Lấy mực nước trước đó
    prev_data = get_previous_river_level(station['location_name'], station['river_name'])
    
    if prev_data:
        prev_level, prev_trend = prev_data
    else:
        # Nếu chưa có dữ liệu, bắt đầu từ mực ngẫu nhiên gần bình thường
        seasonal_adj = get_seasonal_factor()
        prev_level = station['normal_level'] * seasonal_adj * np.random.uniform(0.8, 1.2)
        prev_trend = 'stable'
    
    print(f"  Mực nước trước: {prev_level:.1f}cm (xu hướng: {prev_trend})")
    
    # 1. Yếu tố thời tiết
    weather_impact = get_weather_impact_advanced(weather_data, station)
    
    # 2. Yếu tố mùa vụ
    seasonal_factor = get_seasonal_factor()
    seasonal_impact = (seasonal_factor - 1) * station['normal_level'] * 0.3
    
    # 3. Chu kỳ trong ngày
    daily_cycle = get_daily_cycle_factor()
    daily_impact = (daily_cycle - 1) * station['normal_level'] * 0.1
    
    # 4. Ảnh hưởng thủy triều
    tidal_factor = get_tidal_effect(station)
    tidal_impact = (tidal_factor - 1) * station['normal_level']
    
    # 5. Hoạt động con người
    human_impact = get_human_activities_impact(station)
    
    # 6. Yếu tố địa chất
    geological_impact = get_geological_factors(station)
    
    # 7. Thoát nước tự nhiên
    natural_decline = calculate_natural_flow_change(prev_level, station)
    
    # 8. Xu hướng tiếp tục (quán tính)
    momentum_impact = 0
    if prev_trend == 'rising':
        momentum_impact = np.random.uniform(2, 8)
    elif prev_trend == 'falling':
        momentum_impact = np.random.uniform(-8, -2)
    
    # Tổng hợp tất cả các yếu tố
    total_change = (weather_impact + seasonal_impact + daily_impact + 
                   tidal_impact + human_impact + geological_impact + 
                   momentum_impact - natural_decline)
    
    # Tính mực nước mới
    new_level = prev_level + total_change
    
    # Thêm nhiễu ngẫu nhiên nhỏ (sai số đo lường)
    measurement_noise = np.random.uniform(-2, 2)
    new_level += measurement_noise
    
    # Đảm bảo giới hạn hợp lý
    min_level = station['normal_level'] * 0.2  # Tối thiểu 20% mức bình thường
    max_level = station['alert_level_3'] * 1.3  # Tối đa 130% mức báo động 3
    
    new_level = max(min_level, min(new_level, max_level))
    
    # Tính xu hướng mới
    level_change = new_level - prev_level
    if level_change > 3:
        trend = 'rising'
    elif level_change < -3:
        trend = 'falling'
    else:
        trend = 'stable'
    
    # Tính lưu lượng nước (phức tạp hơn)
    level_ratio = new_level / station['normal_level']
    
    # Lưu lượng tăng theo hàm mũ khi mực nước cao
    flow_rate = station['base_flow_rate'] * (level_ratio ** 1.8)
    
    # Điều chỉnh theo thời tiết
    if weather_data and weather_data.get('rainfall_1h', 0) > 15:
        flow_rate *= 1.3  # Mưa lớn tăng lưu lượng
    
    # Thêm biến động cho lưu lượng
    flow_variation = np.random.uniform(0.85, 1.15)
    flow_rate *= flow_variation
    
    # In chi tiết các yếu tố tác động
    print(f"  Chi tiết tác động:")
    print(f"    - Thời tiết: {weather_impact:+.1f}cm")
    print(f"    - Mùa vụ: {seasonal_impact:+.1f}cm")
    print(f"    - Chu kỳ ngày: {daily_impact:+.1f}cm")
    if station['tidal_effect']:
        print(f"    - Thủy triều: {tidal_impact:+.1f}cm")
    if human_impact != 0:
        print(f"    - Con người: {human_impact:+.1f}cm")
    if geological_impact > 5:
        print(f"    - Địa chất: {geological_impact:+.1f}cm")
    print(f"    - Quán tính: {momentum_impact:+.1f}cm")
    print(f"    - Thoát tự nhiên: -{natural_decline:.1f}cm")
    print(f"    - Tổng thay đổi: {total_change:+.1f}cm")
    
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
    """Lưu dữ liệu mực nước sông vào database"""
    try:
        conn = get_connection()
        if not conn:
            print("Không thể kết nối database")
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
        
        print(f"✓ Đã lưu dữ liệu mực nước cho {station['location_name']} - {station['river_name']}")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu mực nước: {e}")
        return False

def check_duplicate_recent(location_name, river_name, hours=1):
    """Kiểm tra đã có dữ liệu trong vài giờ gần đây chưa"""
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
        print(f"Lỗi khi kiểm tra duplicate: {e}")
        return False

def main():
    print("=== BẮT ĐẦU CRAWL DỮ LIỆU MỰC NƯỚC SÔNG (NÂNG CAO) ===")
    print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Kiểm tra kết nối database
    conn = get_connection()
    if not conn:
        print("❌ Không thể kết nối database. Vui lòng chạy setup_db.py trước")
        return
    else:
        print("✓ Kết nối database thành công")
        close_connection(conn)
    
    success_count = 0
    total_stations = len(RIVER_STATIONS)
    
    # Crawl dữ liệu cho từng trạm đo
    for i, station in enumerate(RIVER_STATIONS, 1):
        print(f"\n[{i}/{total_stations}] Đang xử lý {station['location_name']} - {station['river_name']}...")
        
        # Kiểm tra đã crawl gần đây chưa (trong 1 giờ)
        if check_duplicate_recent(station['location_name'], station['river_name'], 1):
            print(f"⏭️  Đã có dữ liệu gần đây cho {station['location_name']}, bỏ qua...")
            continue
        
        try:
            # Lấy dữ liệu thời tiết tương ứng
            weather_data = get_latest_weather_data(station['location_name'])
            
            if weather_data:
                print(f"  📊 Sử dụng dữ liệu thời tiết thực")
            else:
                print(f"  🎲 Tạo dữ liệu thời tiết mô phỏng")
            
            # Mô phỏng mực nước sông
            river_data = simulate_river_level(station, weather_data)
            
            # Hiển thị thông tin chi tiết
            print(f"  📈 Kết quả:")
            print(f"    Mực nước: {river_data['water_level']:.1f}cm (Bình thường: {station['normal_level']}cm)")
            print(f"    Lưu lượng: {river_data['flow_rate']:.1f}m³/s")
            print(f"    Xu hướng: {river_data['trend']}")
            print(f"    Thay đổi: {river_data['level_change']:+.1f}cm")
            
            # Kiểm tra mức báo động
            alert_level = 0
            if river_data['water_level'] >= station['alert_level_3']:
                print(f"  🚨 CẢNH BÁO CẤP 3: Nguy hiểm! ({river_data['water_level']:.1f}cm >= {station['alert_level_3']}cm)")
                alert_level = 3
            elif river_data['water_level'] >= station['alert_level_2']:
                print(f"  ⚠️  CẢNH BÁO CẤP 2: Cao! ({river_data['water_level']:.1f}cm >= {station['alert_level_2']}cm)")
                alert_level = 2
            elif river_data['water_level'] >= station['alert_level_1']:
                print(f"  ⚡ CẢNH BÁO CẤP 1: Chú ý! ({river_data['water_level']:.1f}cm >= {station['alert_level_1']}cm)")
                alert_level = 1
            else:
                print(f"  ✅ BÌNH THƯỜNG")
            
            # Lưu vào database
            saved = save_river_level_data(station, river_data)
            
            if saved:
                success_count += 1
                print(f"  💾 Lưu thành công")
            else:
                print(f"  ❌ Không thể lưu dữ liệu")
            
        except Exception as e:
            print(f"  ❌ Lỗi xử lý {station['location_name']}: {e}")
        
        # Delay ngẫu nhiên để mô phỏng thực tế
        delay = np.random.uniform(1, 3)
        time.sleep(delay)
    
    print(f"\n=== HOÀN THÀNH CRAWL DỮ LIỆU MỰC NƯỚC SÔNG ===")
    print(f"Thành công: {success_count}/{total_stations} trạm")
    print(f"Thời gian hoàn thành: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
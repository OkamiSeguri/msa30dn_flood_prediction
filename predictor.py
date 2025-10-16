# predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime
from setup_db import get_connection, close_connection

def load_combined_data():
    """Tải dữ liệu kết hợp từ 2 bảng: thời tiết + mực nước sông"""
    try:
        conn = get_connection()
        if not conn:
            print("Không thể kết nối database")
            return None
            
        cursor = conn.cursor()
        
        # Truy vấn kết hợp dữ liệu thời tiết và mực nước
        query = """
        SELECT 
            r.location_name,
            r.latitude, r.longitude,
            r.precipitation,
            r.created_at as weather_time,
            rl.river_name,
            rl.water_level,
            rl.normal_level,
            rl.alert_level_1,
            rl.alert_level_2, 
            rl.alert_level_3,
            rl.flow_rate,
            rl.trend,
            rl.created_at as river_time
        FROM rainfall_data r
        LEFT JOIN river_level_data rl ON r.location_name = rl.location_name
        WHERE rl.water_level IS NOT NULL
        AND DATE(r.created_at) = DATE(rl.created_at)
        ORDER BY r.created_at DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        combined_data = []
        for row in results:
            (location_name, lat, lon, precipitation_json, weather_time, 
             river_name, water_level, normal_level, alert1, alert2, alert3, 
             flow_rate, trend, river_time) = row
            
            # Giải mã dữ liệu thời tiết
            weather_data = json.loads(precipitation_json)
            
            # Chuyển đổi đơn vị nếu cần
            temp = weather_data.get('temperature', 0)
            if temp > 100:
                temp = temp - 273.15
                
            pressure = weather_data.get('pressure', 0)
            if pressure > 50000:
                pressure = pressure / 100
            
            # Tính các chỉ số bổ sung
            water_level_ratio = water_level / normal_level if normal_level > 0 else 1
            flow_rate_normal = flow_rate / 1000 if flow_rate else 0
            
            # Xác định mức báo động hiện tại
            alert_level_exceeded = 0
            if water_level >= alert3:
                alert_level_exceeded = 3
            elif water_level >= alert2:
                alert_level_exceeded = 2
            elif water_level >= alert1:
                alert_level_exceeded = 1
            
            combined_data.append({
                'location_name': location_name,
                'river_name': river_name,
                'latitude': lat,
                'longitude': lon,
                'temperature': float(temp),
                'humidity': float(weather_data.get('humidity', 0)),
                'pressure': float(pressure),
                'rainfall_1h': float(weather_data.get('rainfall_1h', 0)),
                'rainfall_3h': float(weather_data.get('rainfall_3h', 0)),
                'wind_speed': float(weather_data.get('wind_speed', 0)),
                'water_level': float(water_level),
                'water_level_ratio': float(water_level_ratio),
                'normal_level': float(normal_level),
                'alert_level_1': float(alert1),
                'alert_level_2': float(alert2),
                'alert_level_3': float(alert3),
                'flow_rate': float(flow_rate) if flow_rate else 0,
                'flow_rate_normal': float(flow_rate_normal),
                'alert_level_exceeded': int(alert_level_exceeded),
                'trend_rising': 1 if trend == 'rising' else 0,
                'trend_falling': 1 if trend == 'falling' else 0,
                'created_at': weather_time
            })
        
        cursor.close()
        close_connection(conn)
        
        return pd.DataFrame(combined_data)
        
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def load_data_from_db():
    """Tai du lieu tu database (chỉ thời tiết) - để tương thích ngược"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return None
            
        cursor = conn.cursor()
        
        query = """
        SELECT location_name, latitude, longitude, precipitation, created_at
        FROM rainfall_data
        ORDER BY created_at DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        data = []
        for row in results:
            location_name, lat, lon, precipitation_json, created_at = row
            precipitation_data = json.loads(precipitation_json)
            
            # Chuyen nhiet do tu Kelvin sang Celsius neu can
            temp = precipitation_data.get('temperature', 0)
            if temp > 100:
                temp = temp - 273.15
            
            # Chuyen ap suat tu Pa sang hPa neu can
            pressure = precipitation_data.get('pressure', 0)
            if pressure > 50000:
                pressure = pressure / 100
            
            data.append({
                'location_name': location_name,
                'latitude': lat,
                'longitude': lon,
                'temperature': float(temp),
                'humidity': float(precipitation_data.get('humidity', 0)),
                'pressure': float(pressure),
                'rainfall_1h': float(precipitation_data.get('rainfall_1h', 0)),
                'rainfall_3h': float(precipitation_data.get('rainfall_3h', 0)),
                'wind_speed': float(precipitation_data.get('wind_speed', 0)),
                'created_at': created_at
            })
        
        cursor.close()
        close_connection(conn)
        
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Loi khi tai du lieu: {e}")
        return None

def generate_advanced_training_data(real_df):
    """Tạo dữ liệu training nâng cao với 3 cấp độ nguy cơ"""
    
    # Lấy thông tin cơ bản
    if len(real_df) > 0:
        avg_temp = real_df['temperature'].mean()
        avg_humidity = real_df['humidity'].mean()  
        avg_pressure = real_df['pressure'].mean()
        avg_water_level = real_df.get('water_level', pd.Series([200.0])).mean()
    else:
        avg_temp, avg_humidity, avg_pressure = 26.0, 75.0, 1013.0
        avg_water_level = 200.0
    
    synthetic_data = []
    
    # 1. LOW RISK - Rủi ro thấp (60 samples)
    print("Tao du lieu: Nguy co THAP...")
    for i in range(60):
        sample = {
            'location_name': f'Low_Risk_{i}',
            'river_name': f'River_{i%7}',
            'latitude': 10.0 + np.random.uniform(-5, 5),
            'longitude': 106.0 + np.random.uniform(-5, 5),
            'temperature': avg_temp + np.random.uniform(-2, 3),
            'humidity': np.random.uniform(40, 75),
            'pressure': avg_pressure + np.random.uniform(0, 15),
            'rainfall_1h': np.random.uniform(0, 8),
            'rainfall_3h': np.random.uniform(0, 20),
            'wind_speed': np.random.uniform(3, 15),
            'water_level': np.random.uniform(80, 140),
            'water_level_ratio': np.random.uniform(0.6, 0.9),
            'normal_level': 150.0,
            'alert_level_1': 180.0,
            'alert_level_2': 220.0,
            'alert_level_3': 270.0,
            'flow_rate': np.random.uniform(200, 600),
            'flow_rate_normal': np.random.uniform(0.2, 0.6),
            'alert_level_exceeded': 0,
            'trend_rising': np.random.choice([0, 1], p=[0.7, 0.3]),
            'trend_falling': np.random.choice([0, 1], p=[0.5, 0.5]),
            'flood_risk_level': 0  # LOW
        }
        synthetic_data.append(sample)
    
    # 2. MODERATE RISK - Rủi ro trung bình (50 samples)  
    print("Tao du lieu: Nguy co TRUNG BINH...")
    for i in range(50):
        alert_exceeded = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        
        sample = {
            'location_name': f'Moderate_Risk_{i}',
            'river_name': f'River_{i%7}',
            'latitude': 10.0 + np.random.uniform(-5, 5),
            'longitude': 106.0 + np.random.uniform(-5, 5),
            'temperature': avg_temp + np.random.uniform(-3, 1),
            'humidity': np.random.uniform(70, 90),
            'pressure': avg_pressure + np.random.uniform(-10, 5),
            'rainfall_1h': np.random.uniform(5, 18),
            'rainfall_3h': np.random.uniform(15, 40),
            'wind_speed': np.random.uniform(8, 25),
            'water_level': np.random.uniform(160, 240),
            'water_level_ratio': np.random.uniform(0.9, 1.4),
            'normal_level': 150.0,
            'alert_level_1': 180.0,
            'alert_level_2': 220.0,
            'alert_level_3': 270.0,
            'flow_rate': np.random.uniform(500, 1200),
            'flow_rate_normal': np.random.uniform(0.5, 1.2),
            'alert_level_exceeded': alert_exceeded,
            'trend_rising': np.random.choice([0, 1], p=[0.4, 0.6]),
            'trend_falling': np.random.choice([0, 1], p=[0.7, 0.3]),
            'flood_risk_level': 1  # MODERATE
        }
        synthetic_data.append(sample)
    
    # 3. HIGH RISK - Rủi ro cao (40 samples)
    print("Tao du lieu: Nguy co CAO...")
    for i in range(40):
        alert_exceeded = np.random.choice([2, 3], p=[0.4, 0.6])
        
        sample = {
            'location_name': f'High_Risk_{i}',
            'river_name': f'River_{i%7}',
            'latitude': 10.0 + np.random.uniform(-5, 5),
            'longitude': 106.0 + np.random.uniform(-5, 5),
            'temperature': avg_temp + np.random.uniform(-5, -1),
            'humidity': np.random.uniform(85, 99),
            'pressure': np.random.uniform(980, 1005),
            'rainfall_1h': np.random.uniform(15, 50),
            'rainfall_3h': np.random.uniform(35, 100),
            'wind_speed': np.random.uniform(20, 60),
            'water_level': np.random.uniform(240, 320),
            'water_level_ratio': np.random.uniform(1.4, 2.1),
            'normal_level': 150.0,
            'alert_level_1': 180.0,
            'alert_level_2': 220.0,
            'alert_level_3': 270.0,
            'flow_rate': np.random.uniform(1200, 3000),
            'flow_rate_normal': np.random.uniform(1.2, 3.0),
            'alert_level_exceeded': alert_exceeded,
            'trend_rising': np.random.choice([0, 1], p=[0.2, 0.8]),
            'trend_falling': np.random.choice([0, 1], p=[0.9, 0.1]),
            'flood_risk_level': 2  # HIGH
        }
        synthetic_data.append(sample)
    
    print(f"Da tao {len(synthetic_data)} mau du lieu training")
    
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df['created_at'] = pd.Timestamp.now()
    
    return synthetic_df

def create_flood_labels(df):
    """Tạo nhãn nguy cơ lũ 3 cấp độ dựa trên quy tắc thực tế"""
    
    # Kiểm tra có dữ liệu mực nước không
    has_river_data = 'water_level' in df.columns and 'alert_level_exceeded' in df.columns
    
    if has_river_data:
        # Sử dụng quy tắc nâng cao với dữ liệu mực nước
        df['flood_risk_level'] = 0  # Mặc định LOW
        
        # Điều kiện HIGH RISK (2)
        high_conditions = (
            (df['rainfall_1h'] > 20) |
            (df['rainfall_3h'] > 45) |
            (df['alert_level_exceeded'] >= 3) |
            ((df['water_level_ratio'] > 1.5) & (df['trend_rising'] == 1)) |
            ((df['rainfall_1h'] > 15) & (df['alert_level_exceeded'] >= 2)) |
            ((df['humidity'] > 90) & (df['pressure'] < 1000) & (df['rainfall_1h'] > 10))
        )
        
        # Điều kiện MODERATE RISK (1)  
        moderate_conditions = (
            ((df['rainfall_1h'] > 10) & (df['rainfall_1h'] <= 20)) |
            ((df['rainfall_3h'] > 25) & (df['rainfall_3h'] <= 45)) |
            (df['alert_level_exceeded'] == 2) |
            ((df['alert_level_exceeded'] == 1) & (df['trend_rising'] == 1)) |
            ((df['water_level_ratio'] > 1.2) & (df['water_level_ratio'] <= 1.5)) |
            ((df['humidity'] > 85) & (df['rainfall_1h'] > 5)) |
            ((df['pressure'] < 1005) & (df['rainfall_1h'] > 8))
        )
        
        # Áp dụng nhãn
        df.loc[high_conditions, 'flood_risk_level'] = 2
        df.loc[moderate_conditions & (df['flood_risk_level'] != 2), 'flood_risk_level'] = 1
        
    else:
        # Sử dụng quy tắc cơ bản chỉ với dữ liệu thời tiết
        df['flood_risk'] = 0  # Giữ tương thích với code cũ
        
        # Mưa > 15mm/h hoặc > 30mm/3h: nguy cơ cao
        df.loc[(df['rainfall_1h'] > 15) | (df['rainfall_3h'] > 30), 'flood_risk'] = 1
        
        # Độ ẩm cao + mưa vừa: nguy cơ trung bình
        df.loc[(df['humidity'] > 85) & (df['rainfall_1h'] > 8), 'flood_risk'] = 1
        
        # Áp suất thấp + mưa: bão tình
        df.loc[(df['pressure'] < 1005) & (df['rainfall_1h'] > 10), 'flood_risk'] = 1
        
        # Gió mạnh + mưa: tăng nguy cơ
        df.loc[(df['wind_speed'] > 20) & (df['rainfall_1h'] > 5), 'flood_risk'] = 1
        
        # Nếu không có mưa thì vẫn có thể nguy cơ nếu điều kiện xấu
        df.loc[(df['humidity'] > 90) & (df['pressure'] < 1000), 'flood_risk'] = 1
        
        # Chuyển về integer
        df['flood_risk'] = df['flood_risk'].astype(int)
    
    return df

def train_model(df):
    """Huấn luyện mô hình dự báo lũ lụt"""
    if len(df) < 10:
        print(f"Khong du du lieu de huan luyen (can it nhat 10 records, hien co {len(df)})")
        return None, None
    
    print(f"Su dung {len(df)} records de huan luyen")
    
    # Kiểm tra loại dữ liệu
    has_river_data = 'water_level' in df.columns and 'flood_risk_level' in df.columns
    
    if has_river_data:
        # Mô hình nâng cao với dữ liệu mực nước
        features = [
            'temperature', 'humidity', 'pressure', 'rainfall_1h', 'rainfall_3h', 'wind_speed',
            'water_level', 'water_level_ratio', 'flow_rate_normal', 'alert_level_exceeded',
            'trend_rising', 'trend_falling'
        ]
        target = 'flood_risk_level'
        is_advanced = True
    else:
        # Mô hình cơ bản chỉ với dữ liệu thời tiết
        features = ['temperature', 'humidity', 'pressure', 'rainfall_1h', 'rainfall_3h', 'wind_speed']
        target = 'flood_risk'
        is_advanced = False
    
    # Đảm bảo dữ liệu số
    for feature in features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
    
    X = df[features]
    y = df[target].astype(int)
    
    # Thống kê phân bố
    if is_advanced:
        risk_counts = y.value_counts().sort_index()
        print(f"Phan bo nguy co (3 cap do):")
        print(f"  LOW (0): {risk_counts.get(0, 0)} samples")
        print(f"  MODERATE (1): {risk_counts.get(1, 0)} samples") 
        print(f"  HIGH (2): {risk_counts.get(2, 0)} samples")
    else:
        print(f"Phan bo nhan: No flood={sum(y==0)}, Flood={sum(y==1)}")
        print(f"Ti le flood: {sum(y==1)/len(y)*100:.1f}%")
    
    # Chia dữ liệu
    if len(df) > 80 and len(y.unique()) > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    # Huấn luyện mô hình
    if is_advanced:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
    
    try:
        model.fit(X_train, y_train)
        
        # Đánh giá
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Do chinh xac mo hinh: {accuracy:.3f}")
        
        # Báo cáo chi tiết
        if is_advanced:
            target_names = ['LOW', 'MODERATE', 'HIGH']
            print("\nBao cao chi tiet (3 cap do):")
        else:
            target_names = ['No Flood', 'Flood']
            print("\nBao cao chi tiet:")
            
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nMuc do quan trong cua cac yeu to:")
        for _, row in feature_importance.head(8).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return model, features, is_advanced
        
    except Exception as e:
        print(f"Loi khi huan luyen mo hinh: {e}")
        return None, None, False

def predict_flood_risk(model, features, weather_data, is_advanced=False):
    """Dự báo nguy cơ lũ lụt"""
    if model is None:
        return None
    
    try:
        # Tạo DataFrame
        input_data = pd.DataFrame([weather_data])
        
        # Dự báo
        prediction = model.predict(input_data[features])[0]
        probabilities = model.predict_proba(input_data[features])[0]
        
        if is_advanced:
            # Kết quả 3 cấp độ
            risk_labels = ['LOW', 'MODERATE', 'HIGH']
            risk_colors = ['green', 'orange', 'red']
            
            result = {
                'risk_level': risk_labels[prediction],
                'risk_numeric': int(prediction),
                'probabilities': {},
                'confidence': float(max(probabilities)),
                'color': risk_colors[prediction]
            }
            
            # Xử lý probabilities theo số class thực tế
            for i, label in enumerate(risk_labels):
                if i < len(probabilities):
                    result['probabilities'][label] = float(probabilities[i])
                else:
                    result['probabilities'][label] = 0.0
        else:
            # Kết quả 2 cấp độ (tương thích ngược)
            result = {
                'flood_risk': int(prediction),
                'probability_no_flood': float(probabilities[0]),
                'probability_flood': float(probabilities[1]) if len(probabilities) > 1 else 1-probabilities[0],
                'confidence': float(max(probabilities))
            }
        
        return result
        
    except Exception as e:
        print(f"Loi khi du bao: {e}")
        return None

def get_risk_level_text(probability_or_level, is_advanced=False):
    """Chuyển đổi kết quả thành text mô tả"""
    if is_advanced:
        # Đã có sẵn text từ model nâng cao
        return probability_or_level, "auto"
    else:
        # Chuyển đổi xác suất thành mức độ (model cũ)
        if probability_or_level < 0.2:
            return "AN TOAN", "green"
        elif probability_or_level < 0.4:
            return "THAP", "yellow"
        elif probability_or_level < 0.6:
            return "TRUNG BINH", "orange"
        elif probability_or_level < 0.8:
            return "CAO", "red"
        else:
            return "RAT CAO", "darkred"

def save_prediction_result(location_name, prediction_data, input_data, is_advanced=False):
    """Lưu kết quả dự báo vào database"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Tạo khuyến nghị dựa trên kết quả
        if is_advanced:
            risk_level = prediction_data['risk_level']
            if risk_level == 'HIGH':
                recommendations = [
                    "Sơ tán dân cư vùng nguy hiểm",
                    "Chuẩn bị vật tư cứu trợ khẩn cấp", 
                    "Theo dõi mực nước liên tục",
                    "Kích hoạt đội ứng phó khẩn cấp"
                ]
            elif risk_level == 'MODERATE':
                recommendations = [
                    "Theo dõi chặt chẽ diễn biến thời tiết",
                    "Chuẩn bị sẵn sàng biện pháp ứng phó",
                    "Thông báo người dân ở vùng trũng",
                    "Kiểm tra hệ thống thoát nước"
                ]
            else:
                recommendations = [
                    "Tiếp tục theo dõi thông tin thời tiết",
                    "Duy trì hoạt động bình thường"
                ]
            
            probability = prediction_data['probabilities'][risk_level]
        else:
            # Model cũ
            if prediction_data['probability_flood'] > 0.6:
                risk_level = 'HIGH'
                recommendations = ["Cảnh báo nguy cơ lũ cao", "Chuẩn bị biện pháp ứng phó"]
            elif prediction_data['probability_flood'] > 0.4:
                risk_level = 'MODERATE'  
                recommendations = ["Theo dõi tình hình", "Kiểm tra hệ thống thoát nước"]
            else:
                risk_level = 'LOW'
                recommendations = ["Tiếp tục theo dõi"]
            
            probability = prediction_data['probability_flood']
        
        # Kiểm tra bảng có tồn tại không
        cursor.execute("SHOW TABLES LIKE 'flood_predictions'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            query = """
            INSERT INTO flood_predictions 
            (location_name, risk_level, probability, weather_factor, river_factor, 
             combined_score, rainfall_1h, rainfall_3h, water_level, alert_level_exceeded,
             recommendations, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                location_name,
                risk_level,
                probability,
                0.5,  # Weather factor placeholder
                0.5,  # River factor placeholder
                prediction_data.get('confidence', 0.5),
                input_data.get('rainfall_1h', 0),
                input_data.get('rainfall_3h', 0),
                input_data.get('water_level', 0),
                input_data.get('alert_level_exceeded', 0),
                json.dumps(recommendations, ensure_ascii=False),
                'integrated_v1.0'
            )
            
            cursor.execute(query, values)
            conn.commit()
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi luu ket qua du bao: {e}")
        return False

def main():
    print("=== HE THONG DU BAO LU LUT TICH HOP ===")
    print("Tu dong phat hien loai du lieu va chon mo hinh phu hop")
    
    # Thử tải dữ liệu kết hợp trước
    print("Kiem tra du lieu ket hop (thoi tiet + muc nuoc song)...")
    combined_df = load_combined_data()
    
    if combined_df is not None and len(combined_df) > 0:
        print(f"Tim thay {len(combined_df)} records du lieu ket hop")
        print("Su dung mo hinh nang cao voi 3 cap do nguy co")
        
        real_df = combined_df
        use_advanced = True
        
        # Áp dụng nhãn cho dữ liệu thực tế
        real_df = create_flood_labels(real_df)
        
        # Tạo dữ liệu training nâng cao
        synthetic_df = generate_advanced_training_data(real_df)
        
    else:
        print("Khong co du lieu ket hop, su dung du lieu thoi tiet don le")
        print("Su dung mo hinh co ban voi 2 cap do")
        
        # Tải dữ liệu thời tiết cơ bản
        real_df = load_data_from_db()
        if real_df is None:
            real_df = pd.DataFrame()
        
        use_advanced = False
        
        # Áp dụng nhãn cơ bản
        if len(real_df) > 0:
            real_df = create_flood_labels(real_df)
        
        # Tạo dữ liệu training cơ bản (từ code cũ)
        synthetic_data = []
        
        # Tạo dữ liệu mưa to gây lũ
        for i in range(30):
            sample = {
                'location_name': f'Heavy_Rain_{i}',
                'latitude': 10.0 + np.random.uniform(-5, 5),
                'longitude': 106.0 + np.random.uniform(-5, 5),
                'temperature': 26.0 + np.random.uniform(-3, 2),
                'humidity': np.random.uniform(80, 98),
                'pressure': 1013.0 + np.random.uniform(-15, 5),
                'rainfall_1h': np.random.uniform(20, 50),
                'rainfall_3h': np.random.uniform(40, 100),
                'wind_speed': np.random.uniform(15, 35),
                'flood_risk': 1
            }
            synthetic_data.append(sample)
        
        # Tạo dữ liệu không lũ
        for i in range(40):
            sample = {
                'location_name': f'No_Flood_{i}',
                'latitude': 10.0 + np.random.uniform(-5, 5),
                'longitude': 106.0 + np.random.uniform(-5, 5),
                'temperature': 26.0 + np.random.uniform(-2, 3),
                'humidity': np.random.uniform(40, 85),
                'pressure': 1013.0 + np.random.uniform(-5, 15),
                'rainfall_1h': np.random.uniform(0, 12),
                'rainfall_3h': np.random.uniform(0, 25),
                'wind_speed': np.random.uniform(3, 20),
                'flood_risk': 0
            }
            synthetic_data.append(sample)
        
        synthetic_df = pd.DataFrame(synthetic_data)
        synthetic_df['created_at'] = pd.Timestamp.now()
    
    # Kết hợp dữ liệu
    if len(real_df) > 0:
        df = pd.concat([real_df, synthetic_df], ignore_index=True)
        print(f"Ket hop: {len(real_df)} thuc te + {len(synthetic_df)} gia = {len(df)} total")
    else:
        df = synthetic_df
        print(f"Chi su dung du lieu gia: {len(df)} samples")
    
    # Hiển thị thông tin
    print(f"\nThong tin du lieu:")
    print(f"  - Tong so samples: {len(df)}")
    print(f"  - Luong mua TB: {df['rainfall_1h'].mean():.2f}mm/h")
    print(f"  - Do am TB: {df['humidity'].mean():.1f}%")
    print(f"  - Nhiet do TB: {df['temperature'].mean():.1f}°C")
    if 'water_level' in df.columns:
        print(f"  - Muc nuoc TB: {df['water_level'].mean():.1f}cm")
    
    # Huấn luyện mô hình
    result = train_model(df)
    if len(result) == 3:
        model, features, is_advanced = result
    else:
        model, features = result
        is_advanced = use_advanced
    
    if model is not None:
        print(f"\n=== TEST DU BAO ({'NANG CAO' if is_advanced else 'CO BAN'}) ===")
        
        if is_advanced:
            # Test scenarios cho mô hình nâng cao
            test_scenarios = [
                {
                    'name': 'Troi dep - Muc nuoc binh thuong',
                    'temperature': 28, 'humidity': 60, 'pressure': 1015,
                    'rainfall_1h': 0, 'rainfall_3h': 0, 'wind_speed': 8,
                    'water_level': 120, 'water_level_ratio': 0.8, 'flow_rate_normal': 0.4,
                    'alert_level_exceeded': 0, 'trend_rising': 0, 'trend_falling': 0
                },
                {
                    'name': 'Mua vua - Muc nuoc tang',
                    'temperature': 25, 'humidity': 80, 'pressure': 1008,
                    'rainfall_1h': 12, 'rainfall_3h': 28, 'wind_speed': 15,
                    'water_level': 190, 'water_level_ratio': 1.1, 'flow_rate_normal': 0.8,
                    'alert_level_exceeded': 1, 'trend_rising': 1, 'trend_falling': 0
                },
                {
                    'name': 'Mua to - Vuot bao dong cap 2',
                    'temperature': 23, 'humidity': 92, 'pressure': 1002,
                    'rainfall_1h': 25, 'rainfall_3h': 55, 'wind_speed': 25,
                    'water_level': 240, 'water_level_ratio': 1.6, 'flow_rate_normal': 1.5,
                    'alert_level_exceeded': 2, 'trend_rising': 1, 'trend_falling': 0
                }
            ]
        else:
            # Test scenarios cho mô hình cơ bản  
            test_scenarios = [
                {
                    'name': 'Troi nang dep',
                    'temperature': 29.0, 'humidity': 65, 'pressure': 1015,
                    'rainfall_1h': 0, 'rainfall_3h': 0, 'wind_speed': 8
                },
                {
                    'name': 'Mua nhe',
                    'temperature': 27.0, 'humidity': 75, 'pressure': 1012,
                    'rainfall_1h': 4, 'rainfall_3h': 10, 'wind_speed': 12
                },
                {
                    'name': 'Mua to',
                    'temperature': 24.0, 'humidity': 92, 'pressure': 1004,
                    'rainfall_1h': 22, 'rainfall_3h': 50, 'wind_speed': 28
                }
            ]
        
        # Chạy test
        for scenario in test_scenarios:
            name = scenario.pop('name')
            result = predict_flood_risk(model, features, scenario, is_advanced)
            
            if result:
                if is_advanced:
                    print(f"\n{name}:")
                    if 'water_level' in scenario:
                        print(f"  Du lieu: Mua {scenario['rainfall_1h']}mm/h, Muc nuoc {scenario['water_level']}cm")
                    else:
                        print(f"  Du lieu: Mua {scenario['rainfall_1h']}mm/h")
                    print(f"  Ket qua: NGUY CO {result['risk_level']}")
                    print(f"  Xac suat cac cap do:")
                    for level, prob in result['probabilities'].items():
                        print(f"    {level}: {prob:.1%}")
                    print(f"  Do tin cay: {result['confidence']:.1%}")
                    
                    # Cảnh báo
                    if result['risk_level'] == 'HIGH':
                        print(f"  CANH BAO: Nguy co lu cao! Can chuan bi bien phap phong chong")
                    elif result['risk_level'] == 'MODERATE':
                        print(f"  Theo doi: Can theo doi tinh hinh thoi tiet")
                        
                else:
                    risk_text, color = get_risk_level_text(result['probability_flood'], is_advanced)
                    print(f"\n{name}:")
                    print(f"  Du lieu: Mua {scenario['rainfall_1h']:.1f}mm/h")
                    print(f"  Ket qua: NGUY CO {risk_text}")
                    print(f"  Xac suat lu: {result['probability_flood']:.1%}")
                    print(f"  Do tin cay: {result['confidence']:.1%}")
                    
                    # Cảnh báo
                    if result['probability_flood'] > 0.6:
                        print(f"  CANH BAO: Nguy co lu cao! Can chuan bi bien phap phong chong")
                    elif result['probability_flood'] > 0.4:
                        print(f"  Theo doi: Can theo doi tinh hinh thoi tiet")
    
    print(f"\nHoan thanh he thong du bao ({'nang cao' if is_advanced else 'co ban'})")

if __name__ == "__main__":
    main()
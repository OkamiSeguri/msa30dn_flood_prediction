import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONF = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456789"),
    "autocommit": True
}

def create_database():
    """Create the windy_data database if it doesn't exist"""
    try:
        connection = mysql.connector.connect(**DB_CONF)
        cursor = connection.cursor()
        
        cursor.execute("CREATE DATABASE IF NOT EXISTS windy_data")
        print("Database 'windy_data' created successfully")
        
        cursor.close()
        connection.close()
        
    except mysql.connector.Error as err:
        print(f"Error creating database: {err}")
        return False
    
    return True

def create_tables():
    """Create the required tables in windy_data database"""
    try:
        db_conf_with_db = DB_CONF.copy()
        db_conf_with_db['database'] = 'windy_data'
        
        connection = mysql.connector.connect(**db_conf_with_db)
        cursor = connection.cursor()
        
        # Bảng dữ liệu thời tiết (đã có)
        rainfall_table = """
        CREATE TABLE IF NOT EXISTS rainfall_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            location_name VARCHAR(100) NOT NULL,
            latitude DECIMAL(10, 8) NOT NULL,
            longitude DECIMAL(11, 8) NOT NULL,
            precipitation JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_location (location_name),
            INDEX idx_date (created_at)
        )
        """
        
        # Bảng dữ liệu mực nước sông (mới)
        river_level_table = """
        CREATE TABLE IF NOT EXISTS river_level_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            location_name VARCHAR(100) NOT NULL,
            river_name VARCHAR(100) NOT NULL,
            latitude DECIMAL(10, 8) NOT NULL,
            longitude DECIMAL(11, 8) NOT NULL,
            water_level DECIMAL(6, 2) NOT NULL COMMENT 'Mực nước hiện tại (cm)',
            normal_level DECIMAL(6, 2) NOT NULL COMMENT 'Mực nước bình thường (cm)',
            alert_level_1 DECIMAL(6, 2) NOT NULL COMMENT 'Mực nước báo động cấp 1 (cm)',
            alert_level_2 DECIMAL(6, 2) NOT NULL COMMENT 'Mực nước báo động cấp 2 (cm)',
            alert_level_3 DECIMAL(6, 2) NOT NULL COMMENT 'Mực nước báo động cấp 3 (cm)',
            flow_rate DECIMAL(8, 2) COMMENT 'Lưu lượng nước (m3/s)',
            trend VARCHAR(20) COMMENT 'Xu hướng: rising, falling, stable',
            data_source VARCHAR(50) DEFAULT 'simulated',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_location_river (location_name, river_name),
            INDEX idx_date (created_at)
        )
        """
        
        # Bảng dự báo lũ (kết quả)
        flood_prediction_table = """
        CREATE TABLE IF NOT EXISTS flood_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            location_name VARCHAR(100) NOT NULL,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            risk_level ENUM('LOW', 'MODERATE', 'HIGH') NOT NULL,
            probability DECIMAL(5, 4) NOT NULL COMMENT 'Xác suất từ 0.0000 đến 1.0000',
            weather_factor DECIMAL(5, 4) COMMENT 'Ảnh hưởng từ thời tiết',
            river_factor DECIMAL(5, 4) COMMENT 'Ảnh hưởng từ mực nước sông',
            combined_score DECIMAL(5, 4) COMMENT 'Điểm tổng hợp',
            rainfall_1h DECIMAL(6, 2),
            rainfall_3h DECIMAL(6, 2),
            water_level DECIMAL(6, 2),
            alert_level_exceeded INT COMMENT '0=Normal, 1=Alert1, 2=Alert2, 3=Alert3',
            recommendations TEXT COMMENT 'Khuyến nghị hành động',
            model_version VARCHAR(20) DEFAULT 'v1.0',
            INDEX idx_location_time (location_name, prediction_time),
            INDEX idx_risk_level (risk_level)
        )
        """
        
        cursor.execute(rainfall_table)
        cursor.execute(river_level_table)
        cursor.execute(flood_prediction_table)
        
        print("All tables created successfully")
        
        cursor.close()
        connection.close()
        
    except mysql.connector.Error as err:
        print(f"Error creating tables: {err}")
        return False
    
    return True

def get_connection():
    """Get connection to windy_data database"""
    try:
        db_conf_with_db = DB_CONF.copy()
        db_conf_with_db['database'] = 'windy_data'
        connection = mysql.connector.connect(**db_conf_with_db)
        return connection
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def close_connection(conn):
    """Close database connection"""
    if conn and conn.is_connected():
        conn.close()

def test_connection():
    """Test the database connection and show tables"""
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        
        tables = cursor.fetchall()
        print("Tables in windy_data database:")
        for table in tables:
            print(f"  - {table[0]}")
            
        cursor.close()
        close_connection(conn)
        
    except mysql.connector.Error as err:
        print(f"Error testing connection: {err}")
        return False
    
    return True

def setup_database():
    """Complete database setup process"""
    print("Starting database setup for windy_data...")
    print("=" * 50)
    
    if not create_database():
        print("Failed to create database")
        return False
    
    if not create_tables():
        print("Failed to create tables")
        return False
    
    if not test_connection():
        print("Failed to test connection")
        return False
    
    print("=" * 50)
    print("Database setup completed successfully!")
    return True

if __name__ == "__main__":
    setup_database()
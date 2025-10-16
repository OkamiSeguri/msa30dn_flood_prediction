import mysql.connector
from datetime import datetime, timedelta
from setup_db import get_connection, close_connection

def cleanup_old_data(days_to_keep=30):
    """Xoa du lieu cu hon X ngay"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return False
            
        cursor = conn.cursor()
        
        # Tinh ngay gioi han
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Dem so records se bi xoa
        cursor.execute("""
            SELECT COUNT(*) FROM rainfall_data 
            WHERE created_at < %s
        """, (cutoff_date,))
        
        old_count = cursor.fetchone()[0]
        
        if old_count > 0:
            print(f"Se xoa {old_count} records cu hon {days_to_keep} ngay")
            
            # Xoa du lieu cu
            cursor.execute("""
                DELETE FROM rainfall_data 
                WHERE created_at < %s
            """, (cutoff_date,))
            
            conn.commit()
            print(f"Da xoa {old_count} records cu")
        else:
            print("Khong co du lieu cu de xoa")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi xoa du lieu cu: {e}")
        return False

def get_database_stats():
    """Hien thi thong ke database"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return
            
        cursor = conn.cursor()
        
        # Tong so records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_records = cursor.fetchone()[0]
        
        # Records theo ngay
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM rainfall_data 
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 7
        """)
        
        daily_stats = cursor.fetchall()
        
        # Du lieu moi nhat va cu nhat
        cursor.execute("""
            SELECT MIN(created_at), MAX(created_at) 
            FROM rainfall_data
        """)
        
        oldest, newest = cursor.fetchone()
        
        # Kich thuoc database
        cursor.execute("""
            SELECT 
                ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'DB Size in MB'
            FROM information_schema.tables 
            WHERE table_schema = 'windy_data' 
            AND table_name = 'rainfall_data'
        """)
        
        size_result = cursor.fetchone()
        db_size = size_result[0] if size_result else 0
        
        print("=== THONG KE DATABASE ===")
        print(f"Tong so records: {total_records}")
        print(f"Kich thuoc database: {db_size} MB")
        print(f"Du lieu cu nhat: {oldest}")
        print(f"Du lieu moi nhat: {newest}")
        
        print("\nRecords theo ngay (7 ngay gan nhat):")
        for date, count in daily_stats:
            print(f"  {date}: {count} records")
        
        cursor.close()
        close_connection(conn)
        
    except Exception as e:
        print(f"Loi khi lay thong ke: {e}")

def remove_duplicates():
    """Xoa du lieu trung lap"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return False
            
        cursor = conn.cursor()
        
        # Tim duplicates
        cursor.execute("""
            SELECT location_name, latitude, longitude, 
                   DATE(created_at), COUNT(*) as count
            FROM rainfall_data 
            GROUP BY location_name, latitude, longitude, DATE(created_at)
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cursor.fetchall()
        
        if duplicates:
            print(f"Tim thay {len(duplicates)} nhom du lieu trung lap")
            
            # Xoa duplicates, giu lai record moi nhat
            for location, lat, lon, date, count in duplicates:
                cursor.execute("""
                    DELETE t1 FROM rainfall_data t1
                    INNER JOIN rainfall_data t2 
                    WHERE t1.id < t2.id 
                    AND t1.location_name = %s 
                    AND t1.latitude = %s 
                    AND t1.longitude = %s
                    AND DATE(t1.created_at) = %s
                """, (location, lat, lon, date))
                
            conn.commit()
            print("Da xoa cac ban ghi trung lap")
        else:
            print("Khong co du lieu trung lap")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi xoa duplicates: {e}")
        return False

def set_data_retention_limit(max_records=1000):
    """Gioi han so luong records toi da"""
    try:
        conn = get_connection()
        if not conn:
            print("Khong the ket noi database")
            return False
            
        cursor = conn.cursor()
        
        # Dem tong so records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_count = cursor.fetchone()[0]
        
        if total_count > max_records:
            records_to_delete = total_count - max_records
            print(f"Database co {total_count} records, se xoa {records_to_delete} records cu nhat")
            
            # Xoa records cu nhat
            cursor.execute("""
                DELETE FROM rainfall_data 
                ORDER BY created_at ASC 
                LIMIT %s
            """, (records_to_delete,))
            
            conn.commit()
            print(f"Da xoa {records_to_delete} records cu")
        else:
            print(f"Database co {total_count} records, khong can xoa")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Loi khi gioi han du lieu: {e}")
        return False

def main():
    """Menu quan ly database"""
    while True:
        print("\n=== QUAN LY DATABASE ===")
        print("1. Xem thong ke database")
        print("2. Xoa du lieu cu (hon 30 ngay)")
        print("3. Xoa du lieu trung lap")
        print("4. Gioi han records (giu 1000 moi nhat)")
        print("5. Tu dong don dep (cleanup + duplicates + limit)")
        print("0. Thoat")
        
        choice = input("\nChon chuc nang (0-5): ").strip()
        
        if choice == "1":
            get_database_stats()
            
        elif choice == "2":
            days = input("Nhap so ngay muon giu lai (mac dinh 30): ").strip()
            days = int(days) if days.isdigit() else 30
            cleanup_old_data(days)
            
        elif choice == "3":
            remove_duplicates()
            
        elif choice == "4":
            limit = input("Nhap so records toi da (mac dinh 1000): ").strip()
            limit = int(limit) if limit.isdigit() else 1000
            set_data_retention_limit(limit)
            
        elif choice == "5":
            print("Bat dau don dep tu dong...")
            remove_duplicates()
            cleanup_old_data(30)
            set_data_retention_limit(1000)
            print("Hoan thanh don dep tu dong")
            
        elif choice == "0":
            print("Tam biet!")
            break
            
        else:
            print("Lua chon khong hop le!")

if __name__ == "__main__":
    main()
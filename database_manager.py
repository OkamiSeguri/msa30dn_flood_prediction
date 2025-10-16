import mysql.connector
from datetime import datetime, timedelta
from setup_db import get_connection, close_connection

def cleanup_old_data(days_to_keep=30):
    """Delete data older than X days"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return False
            
        cursor = conn.cursor()
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Count records to be deleted
        cursor.execute("""
            SELECT COUNT(*) FROM rainfall_data 
            WHERE created_at < %s
        """, (cutoff_date,))
        
        old_count = cursor.fetchone()[0]
        
        if old_count > 0:
            print(f"Will delete {old_count} records older than {days_to_keep} days")
            
            # Delete old data
            cursor.execute("""
                DELETE FROM rainfall_data 
                WHERE created_at < %s
            """, (cutoff_date,))
            
            conn.commit()
            print(f"Deleted {old_count} old records")
        else:
            print("No old data to delete")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error deleting old data: {e}")
        return False

def get_database_stats():
    """Display database statistics"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return
            
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_records = cursor.fetchone()[0]
        
        # Records by date
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
        
        # Oldest and newest data
        cursor.execute("""
            SELECT MIN(created_at), MAX(created_at) 
            FROM rainfall_data
        """)
        
        oldest, newest = cursor.fetchone()
        
        # Database size
        cursor.execute("""
            SELECT 
                ROUND(((data_length + index_length) / 1024 / 1024), 2) AS 'DB Size in MB'
            FROM information_schema.tables 
            WHERE table_schema = 'windy_data' 
            AND table_name = 'rainfall_data'
        """)
        
        size_result = cursor.fetchone()
        db_size = size_result[0] if size_result else 0
        
        print("=== DATABASE STATISTICS ===")
        print(f"Total records: {total_records}")
        print(f"Database size: {db_size} MB")
        print(f"Oldest data: {oldest}")
        print(f"Newest data: {newest}")
        
        print("\nRecords by date (last 7 days):")
        for date, count in daily_stats:
            print(f"  {date}: {count} records")
        
        cursor.close()
        close_connection(conn)
        
    except Exception as e:
        print(f"Error fetching statistics: {e}")

def remove_duplicates():
    """Delete duplicate data"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return False
            
        cursor = conn.cursor()
        
        # Find duplicates
        cursor.execute("""
            SELECT location_name, latitude, longitude, 
                   DATE(created_at), COUNT(*) as count
            FROM rainfall_data 
            GROUP BY location_name, latitude, longitude, DATE(created_at)
            HAVING COUNT(*) > 1
        """)
        
        duplicates = cursor.fetchall()
        
        if duplicates:
            print(f"Found {len(duplicates)} groups of duplicate data")
            
            # Delete duplicates, keep the latest record
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
            print("Deleted duplicate records")
        else:
            print("No duplicate data found")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error deleting duplicates: {e}")
        return False

def set_data_retention_limit(max_records=1000):
    """Set maximum record limit"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return False
            
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        total_count = cursor.fetchone()[0]
        
        if total_count > max_records:
            records_to_delete = total_count - max_records
            print(f"Database has {total_count} records, will delete {records_to_delete} oldest records")
            
            # Delete oldest records
            cursor.execute("""
                DELETE FROM rainfall_data 
                ORDER BY created_at ASC 
                LIMIT %s
            """, (records_to_delete,))
            
            conn.commit()
            print(f"Deleted {records_to_delete} old records")
        else:
            print(f"Database has {total_count} records, no deletion needed")
        
        cursor.close()
        close_connection(conn)
        return True
        
    except Exception as e:
        print(f"Error setting data limit: {e}")
        return False

def main():
    """Database management menu"""
    while True:
        print("\n=== DATABASE MANAGEMENT ===")
        print("1. View database statistics")
        print("2. Delete old data (older than 30 days)")
        print("3. Delete duplicate data")
        print("4. Limit records (keep 1000 most recent)")
        print("5. Auto cleanup (cleanup + duplicates + limit)")
        print("0. Exit")
        
        choice = input("\nSelect function (0-5): ").strip()
        
        if choice == "1":
            get_database_stats()
            
        elif choice == "2":
            days = input("Enter number of days to keep (default 30): ").strip()
            days = int(days) if days.isdigit() else 30
            cleanup_old_data(days)
            
        elif choice == "3":
            remove_duplicates()
            
        elif choice == "4":
            limit = input("Enter maximum records (default 1000): ").strip()
            limit = int(limit) if limit.isdigit() else 1000
            set_data_retention_limit(limit)
            
        elif choice == "5":
            print("Starting auto cleanup...")
            remove_duplicates()
            cleanup_old_data(30)
            set_data_retention_limit(1000)
            print("Auto cleanup completed")
            
        elif choice == "0":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
from setup_db import get_connection, close_connection

def check_data():
    """Check data in database"""
    try:
        conn = get_connection()
        if not conn:
            print("Cannot connect to database")
            return
            
        cursor = conn.cursor()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM rainfall_data")
        count = cursor.fetchone()[0]
        print(f"Total records: {count}")
        
        # Fetch the 5 most recent records
        cursor.execute("""
        SELECT location_name, latitude, longitude, created_at 
        FROM rainfall_data 
        ORDER BY created_at DESC 
        LIMIT 5
        """)
        
        records = cursor.fetchall()
        print("\n5 most recent records:")
        for record in records:
            print(f"   - {record[0]}: ({record[1]}, {record[2]}) - {record[3]}")
        
        cursor.close()
        close_connection(conn)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()
import mysql.connector
import yaml
import os

def get_db_connection():
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        db = mysql.connector.connect(
            host=config['mysql']['host'],
            user=config['mysql']['user'],
            password=config['mysql']['password'],
            database=config['mysql']['database']
        )
        print("✅ MySQL connection established.")
        return db
    except mysql.connector.Error as err:
        print(f"❌ MySQL connection error: {err}")
        return None

def create_table_if_not_exists(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS entry_exit_log (
        id INT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME,
        event VARCHAR(10),
        count_change INT
    )
    """)
    cursor._connection.commit()

if __name__ == "__main__":
    db = get_db_connection()
    if db:
        cursor = db.cursor()
        create_table_if_not_exists(cursor)
        print("✅ Table created successfully.")
        db.close()
    else:
        print("❌ Failed to connect to the database.")

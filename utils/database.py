import mysql.connector
import yaml
import os


def get_db_connection():
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    mysql_cfg = config.get('mysql', {})
    host = os.environ.get('MYSQL_HOST', mysql_cfg.get('host', 'localhost'))
    user = os.environ.get('MYSQL_USER', mysql_cfg.get('user', 'root'))
    password = os.environ.get('MYSQL_PASSWORD', mysql_cfg.get('password', ''))
    database = os.environ.get('MYSQL_DATABASE', mysql_cfg.get('database', 'people_control'))

    try:
        db = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            connection_timeout=5
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

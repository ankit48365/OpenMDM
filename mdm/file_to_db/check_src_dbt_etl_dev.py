import sqlite3

conn = sqlite3.connect(r'D:/mygit/OpenMDM/database/dbt_etl_dev.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='personal_info'")
result = cursor.fetchone()

if result:
    print("✅ Table 'personal_info' exists.")
else:
    print("❌ Table 'personal_info' not found.")

conn.close()
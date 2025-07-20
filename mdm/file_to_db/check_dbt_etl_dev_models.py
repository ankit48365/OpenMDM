import sqlite3  # standard python package , does not need install via uv or pip
import csv  # standard python package , does not need install via uv or pip
print("sqlite version : ",sqlite3.sqlite_version)

## Create DB

conn = sqlite3.connect(r'D:\mygit\OpenMDM\database\dbt_etl_dev_models.db')
cursor = conn.cursor()



from tabulate import tabulate

# # Count total records
# cursor.execute('SELECT COUNT(*) FROM personal_info')
# total_records = cursor.fetchone()[0]
# print(f"\n📊 Total records: {total_records}\n")

# Fetch and display schema name(s)
cursor.execute("SELECT name FROM pragma_database_list")
schema_rows = cursor.fetchall()
schema_headers = ['Schema']
print("\n🗂️ Connected Schemas:\n")
print(tabulate(schema_rows, headers=schema_headers, tablefmt='grid'))



# Fetch first few records
cursor.execute('SELECT * FROM main.personal_info LIMIT 3')
rows = cursor.fetchall()

# Pretty print as a table
headers = [description[0] for description in cursor.description]
print(tabulate(rows, headers=headers, tablefmt='grid'))

conn.commit()
conn.close()
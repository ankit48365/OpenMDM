import sqlite3  # standard python package , does not need install via uv or pip
import csv  # standard python package , does not need install via uv or pip
print("sqlite version : ",sqlite3.sqlite_version)

## Create DB

conn = sqlite3.connect(r'D:\mygit\OpenMDM\database\dbt_etl_dev.db')
cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS personal_info')

cursor.execute('''
CREATE TABLE personal_info (
    first_name TEXT,
    middle_name TEXT,
    last_name TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    phone TEXT,
    email TEXT,
    original INTEGER,
    COMDL INTEGER,
    REAL INTEGER
)
''')


with open(r'D:\mygit\OpenMDM\mdm\source_data\Data_Set_2.csv', newline='', encoding='utf-8') as csvfile:

    reader = csv.DictReader(csvfile) # Using DictReader for field by column names and fill missing values with empty string
    for row in reader:
        values = [row.get(col, '') for col in [
            'first_name', 'middle_name', 'last_name', 'address', 'city',
            'state', 'zip_code', 'phone', 'email', 'original', 'COMDL', 'REAL'
        ]]
        cursor.execute('INSERT INTO personal_info VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', values)


from tabulate import tabulate

# Count total records
cursor.execute('SELECT COUNT(*) FROM personal_info')
total_records = cursor.fetchone()[0]
print(f"\n📊 Total records: {total_records}\n")

# Fetch and display schema name(s)
cursor.execute("SELECT name FROM pragma_database_list")
schema_rows = cursor.fetchall()
schema_headers = ['Schema']
print("\n🗂️ Connected Schemas:\n")
print(tabulate(schema_rows, headers=schema_headers, tablefmt='grid'))



# Fetch first few records
cursor.execute('SELECT * FROM personal_info LIMIT 3')
rows = cursor.fetchall()

# Pretty print as a table
headers = [description[0] for description in cursor.description]
print(tabulate(rows, headers=headers, tablefmt='grid'))

conn.commit()
conn.close()

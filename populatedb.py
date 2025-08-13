import os
import sqlite3
import pandas as pd

def create_and_populate_db():
    # Ensure DB folder exists
    os.makedirs("db", exist_ok=True)

    # Connect to SQLite DB
    conn = sqlite3.connect("db/stock_data.db")
    cursor = conn.cursor()

    # Prepare table schema if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            Dividends REAL,
            Stock_Splits REAL,
            Stock TEXT
        )
    """)
    conn.commit()

    # Read CSVs from folder and insert
    for filename in os.listdir("data/day"):
        if filename.endswith(".csv"):
            try:
                file_path = os.path.join("data/day", filename)
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # Clean column names

                # Add Stock column based on filename
                stock_name = filename.replace(".csv", "").replace(".CSV", "").strip()
                df["Stock"] = stock_name

                # Select and rename columns to match schema (optional if consistent)
                df = df.rename(columns={
                    "Stock Splits": "Stock_Splits"
                })

                # Append to single table
                df.to_sql("stock_data", conn, if_exists="append", index=False)
                print(f"✅ Loaded {filename} ({len(df)} rows)")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    conn.commit()
    conn.close()
    print("📦 Done populating stock_data.db")

# Run the function
if __name__ == "__main__":
    create_and_populate_db()

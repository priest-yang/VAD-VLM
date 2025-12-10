import os
import sqlite3
import pickle
import argparse
from tqdm import tqdm

def convert_pkls_to_sqlite(pkl_path, sqlite_db_path):
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Create the 'infos' table with additional columns for timestamp, location, and scene
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS infos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data BLOB NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            location TEXT NOT NULL,
            scene TEXT NOT NULL, 
            frame_idx INTEGER NOT NULL, 
            lidar_pc CHAR(18) NOT NULL,
            UNIQUE(timestamp, location, scene) ON CONFLICT IGNORE
        )
    ''')
    conn.commit()

    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
        if "infos" in pkl_data:
            infos = pkl_data["infos"]
        else:
            infos = pkl_data
        
        # Insert the data into the SQLite table
        for info in tqdm(infos, total=len(infos)):
            serialized_info = pickle.dumps(info)
            timestamp = info['timestamp']
            location = info['map_location']
            scene = info['scene_file'][:-3]
            frame_idx = info['frame_idx']
            lidar_pc = info['lidar_pc']
            # Insert into the database, ignoring duplicates based on timestamp, location, and scene
            cursor.execute('''
                INSERT OR IGNORE INTO infos (data, timestamp, location, scene, frame_idx, lidar_pc) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (serialized_info, timestamp, location, scene, frame_idx, lidar_pc))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert a pickle file to an SQLite database")
    parser.add_argument("pkl_path", help="Path to the input pickle file")
    parser.add_argument("sqlite_db_path", help="Path to the SQLite database file")

    args = parser.parse_args()

    # Call the main function with the arguments
    convert_pkls_to_sqlite(args.pkl_path, args.sqlite_db_path)



########################################
# play with the sqlite if you may want #
########################################

# import sqlite3

# # Open a connection to the SQLite database
# conn = sqlite3.connect('mydb.db')
# cursor = conn.cursor()

# # Execute the query
# cursor.execute('''
#     SELECT id, location, timestamp, scene, frame_idx, lidar_pc
#     FROM infos
#     ORDER BY scene, timestamp
# ''')

# # Fetch all the results
# results = cursor.fetchall()

# # Close the connection
# conn.close()

# # Print the results
# for row in results:
#     print(row)

import math
import sqlite3
import math

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('activations.db')
cursor = conn.cursor()

# Function to save activations
def save_activations(url, activation_values):
    data = [(url, i, activation_values[i]) for i in range(len(activation_values))]
    cursor.executemany("INSERT INTO activations (url, activation_index, activation_value) VALUES (?, ?, ?)", data)
    conn.commit()

# Create table with dynamic columns based on num_latents
def setup():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activations (
            id INTEGER PRIMARY KEY,
            url TEXT,
            activation_index INTEGER,
            activation_value REAL
        )
    """)
    conn.commit()

def top_k(activation_index, k):
    query = """
        SELECT url, activation_value 
        FROM activations 
        WHERE activation_index = ? 
        ORDER BY activation_value DESC 
        LIMIT ?
    """
    cursor.execute(query, (activation_index, k))
    results = cursor.fetchall()
    return results

def log_sparsity(activation_index, threshold=0):
    query = """
        SELECT COUNT(*) 
        FROM activations 
        WHERE activation_index = ? 
        AND activation_value > ?
    """
    cursor.execute(query, (activation_index, threshold))
    count_above_threshold = cursor.fetchone()[0]

    query = "SELECT COUNT(*) FROM activations WHERE activation_index = ?"
    cursor.execute(query, (activation_index,))
    total_count = cursor.fetchone()[0]

    if total_count == 0:
        return -math.inf  # Log of zero, handle case with no data

    percent_above_threshold = count_above_threshold / total_count
    if percent_above_threshold == 0:
        return -math.inf  # Log of zero, handle case with no activations above threshold

    return math.log10(percent_above_threshold)

def mean_k(activation_index):
    query = "SELECT AVG(activation_value) FROM activations WHERE activation_index = ?"
    cursor.execute(query, (activation_index,))
    mean_value = cursor.fetchone()[0]
    return mean_value


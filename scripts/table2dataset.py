"""
Convert a table to a dataset format.
"""
import os
import datasets
import mysql.connector



def get_table():
    """
    Get all files with status 'success' for a specific provider from the database

    Returns:
        list: List of files with status 'success'
    """
    # Database connection parameters
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")

    db_config = {
        'host': db_host,
        'user': db_user,
        'password': db_password,
        'database': db_name
    }
    try:
        # Establish database connection
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Execute query with parameter binding for security
        query = """
                  SELECT question, answer, context
                  FROM llama_rag
                  """
        cursor.execute(query)

        # Fetch all matching records
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return results

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        return []

def main():
    """
    Main function to execute the script.
    """
    # Get the table
    table = get_table()

    datasets
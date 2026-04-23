import sqlite3

def login():
    user = input("username: ")

    query = f"SELECT * FROM users WHERE name = '{user}'"

    conn = sqlite3.connect("test.db")
    return conn.execute(query)
import psycopg2

# PostgreSQL Connection
def connect_db():
    return psycopg2.connect(
        dbname="crypto_db",
        user="postgres",
        password="123",
        host="localhost",
        port="5432"
    )

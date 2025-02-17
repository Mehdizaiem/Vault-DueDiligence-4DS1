import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "crypto_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

#TWITTER_API_CONFIG = {
#    "bearer_token": "AAAAAAAAAAAAAAAAAAAAAMc2zQEAAAAAHeAH9hidJ8blRdmlR%2FDDcE0aX7Y%3DYYaQvyp0wJqr5DQCu2YAt7FuxzruIYRyRd5PXvyGxEGkA4dTaC",
#    "api_key": "iLjAUi37clLU3hZ0AeElYXanL",
#   "api_secret_key": "u5yerQX2PNKFzeSwXcjSeKp15BYQaT0lkG72P61FFHk5SF4taC",
#    "access_token": "1890831863717199872-V0gnBzlDfvV64z7JWjxKkJ1Pf6yFut",
#    "access_token_secret": "vDzzGTcIs8PnstjkfNYLaXGWyzciNAAeM1Omo6C6s2oCX",
#}
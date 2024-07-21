from sqlalchemy import create_engine
import pandas as pd
import os

class DB:
    def __init__(self):
        self.host = os.getenv("PG_HOST")
        self.port = os.getenv("PG_PORT")
        self.dbname = os.getenv("PG_DATABASE")
        self.user = os.getenv("PG_USERNAME")
        self.password = os.getenv("PG_PASSWORD")
        self.engine = None

        try:
            self.engine = create_engine(
                f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'
            )
            print("Connection to the PostgreSQL database was successful.")
        except Exception as e:
            print(f"Error connecting to the PostgreSQL database: {e}")

    def get_historical_data(self):

        query = "SELECT * FROM historical_data"
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"Error fetching data from historical_data table: {e}")
            return None

    def __del__(self):
        if self.engine:
            self.engine.dispose()
            print("Database connection closed.")

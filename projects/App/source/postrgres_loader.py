import os
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from sqlalchemy.engine import Connection

# todo move to env
DATA = {
    "apple_store": "./data/apple_data/AppleStore.csv",
    "apple_descriptions": "./data/apple_data/AppleStore_description.csv",
    "google_store": "./data/google_data/googleplaystore.csv",
    "google_reviews": "./data/google_data/googleplaystore_user_reviews.csv",
}


def load_table(connection: Connection, table_name: str, file_name: str) -> None:
    df = pd.read_csv(file_name)
    df.to_sql(table_name, con=conn, if_exists='replace', index=False)


if __name__ == '__main__':
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    conn_string = 'postgresql://postgres:chinois1@localhost/postgres'
    db = create_engine(conn_string)
    conn = db.connect()

    for k, v in DATA.items():
        print(f"Load table {k} from {v}")
        load_table(conn, k, v)
    conn.close()

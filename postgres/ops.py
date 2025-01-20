import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

class PostgresInterface:
    """
    A simple interface for connecting to a PostgreSQL database and interacting
    with it using pandas DataFrames. Reads credentials from a .env file.
    Uses SQLAlchemy for more flexible SQL execution.
    """

    def __init__(self, env_path=".env"):
        """
        Initializes the interface, reading database connection parameters from
        a .env file.

        Args:
            env_path (str, optional): The path to the .env file. Defaults to ".env".
        """
        load_dotenv(dotenv_path=env_path)

        self.dbname = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.engine = None

        if not all([self.dbname, self.user, self.password]):
            raise ValueError(
                "Missing database credentials (DB_NAME, DB_USER, or DB_PASSWORD) in .env file."
            )

    def connect(self):
        """
        Establishes a connection to the PostgreSQL database using SQLAlchemy.
        """
        try:
            db_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            self.engine = create_engine(db_url)
            print("Successfully connected to the database!")
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    def disconnect(self):
        """
        Closes the database connection.
        """
        if self.engine:
            self.engine.dispose()
            print("Database connection closed.")

    def execute_sql(self, sql_query, params=None):
        """
        Executes an arbitrary SQL query and returns the results as a pandas DataFrame.

        Args:
            sql_query (str): The SQL query to execute.
            params (dict, optional): A dictionary of parameters to pass to the query. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing the query results, or None if an error occurred.
        """
        if not self.engine:
            print("Not connected to the database. Please connect first.")
            return None

        try:
            with self.engine.connect() as connection:
                if params:
                    result = connection.execute(text(sql_query), params)
                else:
                    result = connection.execute(text(sql_query))

                # Fetch all results and convert to DataFrame if there are any rows
                rows = result.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=result.keys())
                    return df
                else:
                    print("Query returned no rows.")
                    return pd.DataFrame()  # Return an empty DataFrame
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def write_df_to_table(self, df, table_name, if_exists="fail", index=False):
        """
        Writes a pandas DataFrame to a table in the database using df.to_sql().

        Args:
            df (pandas.DataFrame): The DataFrame to write.
            table_name (str): The name of the table to write to.
            if_exists (str, optional): Specifies how to behave if the table already exists.
                - 'fail': Raise a ValueError.
                - 'replace': Drop the table before inserting new values.
                - 'append': Insert new values after existing ones.
                Defaults to 'fail'.
            index (bool, optional): Write DataFrame index as a column. Defaults to False.
        """
        if not self.engine:
            print("Not connected to the database. Please connect first.")
            return

        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=index
            )
            print(f"Successfully wrote DataFrame to table '{table_name}'.")
        except Exception as e:
            print(f"Error writing DataFrame to table: {e}")

# Example Usage:
if __name__ == "__main__":
    # Create an instance of the interface
    pg_interface = PostgresInterface(env_path=".env")

    # Connect to the database
    pg_interface.connect()

    # Example: Execute an arbitrary SQL query with parameters
    query = "SELECT * FROM your_table_name WHERE column1 > :value" # Replace with your query
    result_df = pg_interface.execute_sql(query)
    if result_df is not None:
        print("Query Result shape:")
        print(result_df.shape)

    # Example: Write a DataFrame to a table
    # pg_interface.write_df_to_table(new_df, "new_table", if_exists="append")

    # Disconnect from the database
    pg_interface.disconnect()
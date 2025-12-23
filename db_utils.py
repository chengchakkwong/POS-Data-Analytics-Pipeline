import os
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 加載環境變數
load_dotenv()

class POSDatabaseManager:
    """
    【基礎建設層】
    專門負責資料庫的連線與最底層的 SQL 執行。
    """
    def __init__(self, timeout=5):
        self.driver = os.getenv('DB_DRIVER', '{ODBC Driver 17 for SQL Server}')
        self.server = os.getenv('DB_SERVER')
        self.database = os.getenv('DB_DATABASE')
        self.uid = os.getenv('DB_UID')
        self.pwd = os.getenv('DB_PWD')
        self.trust_cert = os.getenv('DB_TRUST_CERT', 'yes')
        self.timeout = timeout
        self.engine = self._create_engine()

    def _create_engine(self):
        conn_str = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.uid};"
            f"PWD={self.pwd};"
            f"TrustServerCertificate={self.trust_cert};"
            f"LoginTimeout={self.timeout};"
        )
        quoted_params = urllib.parse.quote_plus(conn_str)
        return create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_params}")

    def execute_query(self, sql, params=None):
        import pandas as pd
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn, params=params)
        except Exception as e:
            print(f"❌ 查詢失敗: {e}")
            return pd.DataFrame()

    def check_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
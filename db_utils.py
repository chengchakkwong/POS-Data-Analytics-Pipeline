import os
import logging
import urllib.parse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logger_config  # 導入統一的日誌配置

# 加載環境變數
load_dotenv()

# 使用統一的日誌配置
logger = logging.getLogger(__name__)

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

    def _sanitize_error(self, error):
        """安全地處理錯誤信息，移除可能包含敏感資料的部分"""
        error_str = str(error)
        # 移除可能包含連接字符串的部分
        if 'DRIVER=' in error_str or 'SERVER=' in error_str:
            return "資料庫連接錯誤（詳細信息已隱藏）"
        # 移除可能包含密碼的部分
        if 'PWD=' in error_str or 'password' in error_str.lower():
            return "資料庫認證錯誤（詳細信息已隱藏）"
        # 移除可能包含用戶名的部分
        if 'UID=' in error_str or 'user' in error_str.lower():
            return "資料庫認證錯誤（詳細信息已隱藏）"
        # 只返回錯誤類型，不返回詳細信息
        return f"資料庫操作失敗: {type(error).__name__}"

    def execute_query(self, sql, params=None):
        import pandas as pd
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn, params=params)
        except Exception as e:
            # 安全：詳細錯誤記錄到日誌文件（AI 無法讀取）
            # 使用 exc_info=True 記錄完整的堆棧追蹤
            logger.error(f"查詢失敗: {e}", exc_info=True)
            return pd.DataFrame()

    def check_connection(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            # 安全：詳細錯誤記錄到日誌文件，不輸出到終端
            logger.error(f"連接檢查失敗: {e}", exc_info=True)
            return False
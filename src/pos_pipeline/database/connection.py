"""Database connection helpers."""

from __future__ import annotations

import os
import urllib.parse
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

REQUIRED_DB_ENV = (
    "DB_SERVER",
    "DB_DATABASE",
    "DB_UID",
    "DB_PWD",
)


def get_missing_db_env() -> list[str]:
    """Return names of required DB env vars that are missing."""
    missing = []
    for name in REQUIRED_DB_ENV:
        value = os.getenv(name)
        if value is None or str(value).strip() == "":
            missing.append(name)
    return missing


def create_engine_from_env(timeout: int = 5) -> Engine:
    """Create a SQLAlchemy engine from environment variables."""
    missing = get_missing_db_env()
    if missing:
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")

    driver = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
    trust_cert = os.getenv("DB_TRUST_CERT", "yes")

    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_DATABASE')};"
        f"UID={os.getenv('DB_UID')};"
        f"PWD={os.getenv('DB_PWD')};"
        f"TrustServerCertificate={trust_cert};"
        f"LoginTimeout={timeout};"
    )
    quoted = urllib.parse.quote_plus(conn_str)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")


def check_connection(timeout: int = 5) -> bool:
    """Return True if SELECT 1 succeeds."""
    engine = create_engine_from_env(timeout=timeout)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def execute_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    engine = create_engine_from_env()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
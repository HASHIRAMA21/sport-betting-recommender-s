import urllib
from typing import Optional, Dict, Any

import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

from helper.logging import logger
from models.model_config import AlgorithmConfig


class DatabaseConnector:
    """Gestionnaire de connexion à la base de données MySQL."""

    def __init__(self, config: AlgorithmConfig):
        """Initialise la connexion avec la configuration."""
        self.config = config
        self.connection = None
        self.engine = None

        # Construction de la chaîne de connexion
        encoded_password = urllib.parse.quote_plus(config.db_password)
        self.connection_string = (
            f"mysql+pymysql://{config.db_user}:{encoded_password}@"
            f"{config.db_host}:{config.db_port}/{config.db_name}"
        )

    def connect(self) -> None:
        """Établit la connexion à la base de données."""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
                charset='utf8mb4',
                autocommit=True
            )

            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600
            )

            logger.info("Connexion à la base de données établie")

        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            raise

    def disconnect(self) -> None:
        """Ferme la connexion."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Connexion DB fermée")

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Exécute une requête et retourne un DataFrame."""
        try:
            if params:
                return pd.read_sql(query, self.engine, params=params)
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Erreur requête SQL: {e}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Retourne les informations d'une table."""
        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = '{self.config.db_name}' 
        AND TABLE_NAME = '{table_name}'
        """
        return self.execute_query(query)
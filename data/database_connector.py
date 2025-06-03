import urllib

import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

from helper.logging import logger

password = "Sport@Bet19"
encoded_password = urllib.parse.quote_plus(password)
connection_string = f"mysql+pymysql://sport_bet:{encoded_password}@146.59.148.113:51724/ai_engine_db"


class DatabaseConnector:
    """Classe pour gérer les connexions à la base de données MySQL."""

    def __init__(self, host: str, port: str, user: str, password: str, database: str):
        """
        Initialise la connexion à la base de données.

        Args:
            host: Hôte de la base de données
            user: Nom d'utilisateur
            password: Mot de passe
            database: Nom de la base de données
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.engine = None

    def connect(self) -> None:
        """Établit la connexion à la base de données."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.engine = create_engine(f"mysql+pymysql://sport_bet:{encoded_password}@146.59.148.113:51724/ai_engine_db") #create_engine(f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
            logger.info("Connexion à la base de données établie avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de la connexion à la base de données: {e}")
            raise

    def disconnect(self) -> None:
        """Ferme la connexion à la base de données."""
        if self.connection:
            self.connection.close()
            logger.info("Connexion à la base de données fermée.")

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Exécute une requête SQL et retourne les résultats sous forme de DataFrame.

        Args:
            query: Requête SQL à exécuter

        Returns:
            DataFrame contenant les résultats de la requête
        """
        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête: {e}")
            raise

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """
        Insère un DataFrame dans une table de la base de données.

        Args:
            df: DataFrame à insérer
            table_name: Nom de la table
            if_exists: Comportement si la table existe déjà ('replace', 'append', 'fail')
        """
        try:
            df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=False)
            logger.info(f"DataFrame inséré avec succès dans la table {table_name}.")
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion du DataFrame: {e}")
            raise

from typing import Optional, Dict

import pandas as pd

from data.database_connector import DatabaseConnector
from helper.logging import logger


class SportsDataLoader:
    """Chargeur de données spécialisé pour les paris sportifs."""

    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        self.data_cache = {}

    def load_users_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des utilisateurs."""
        query = """
                SELECT id AS user_id,
                       registration_date,
                       country,
                       age,
                       total_deposits,
                       total_withdrawals,
                       vip_status,
                       last_login_date,
                       DATEDIFF(CURDATE(), registration_date) as days_since_registration
                FROM t_etl_user_summary
                """
        #!-- WHERE status = 'active'

        if limit:
            query += f" LIMIT {limit}"

        logger.info("Chargement des données utilisateurs...")
        return self.db.execute_query(query)

    def load_events_data(self, days_back: int = 30, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des événements sportifs."""
        query = """
                SELECT id AS event_id,
                       event_type_name                           AS sport,
                       competition_name                     AS competition,
                       home_team,
                       away_team,
                       CONCAT(home_team, ' vs ', away_team) AS teams,
                       event_start_time,
                       event_status,
                       venue,
                       country_name AS country,
                       CASE
                           WHEN event_start_time > NOW() THEN TIMESTAMPDIFF(HOUR, NOW(), event_start_time)
                           ELSE 0
                           END                              AS hours_until_event
                FROM t_etl_event_summary
                WHERE event_start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
                  AND event_status IN ('scheduled', 'live', 'finished') \
                """

        if limit:
            query += " LIMIT %s"
            params = (days_back, limit)
        else:
            params = (days_back,)

        logger.info(f"Chargement des événements ({days_back} derniers jours)...")
        return self.db.execute_query(query, params)

    def load_markets_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des marchés de paris."""
        query = """
                SELECT market_id, \
                       event_id, \
                       market_name, \
                       market_type, \
                       status     as market_status, \
                       created_at as market_open_time
                FROM markets
                WHERE status IN ('open', 'suspended', 'closed') \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("Chargement des marchés...")
        return self.db.execute_query(query)

    def load_outcomes_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des outcomes."""
        query = """
                SELECT outcome_id, \
                       market_id, \
                       outcome_name, \
                       current_odds, \
                       status as outcome_status
                FROM outcomes
                WHERE status IN ('active', 'suspended', 'settled') \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("Chargement des outcomes...")
        return self.db.execute_query(query)

    def load_bets_data(self, days_back: int = 90, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des paris."""
        query = """
                SELECT bet_id, \
                       user_id, \
                       outcome_id, \
                       bet_amount, \
                       odds_used, \
                       bet_timestamp, \
                       settlement_timestamp, \
                       status  as bet_status, \
                       CASE \
                           WHEN status = 'won' THEN 1 \
                           WHEN status = 'lost' THEN 0 \
                           ELSE NULL \
                           END as outcome, \
                       CASE \
                           WHEN TIMESTAMPDIFF(MINUTE, bet_timestamp, \
                                                      (SELECT event_start_time \
                                                       FROM events e \
                                                                JOIN markets m ON e.event_id = m.event_id \
                                                                JOIN outcomes o ON m.market_id = o.market_id \
                                                       WHERE o.outcome_id = bets.outcome_id) \
                                ) <= 0 THEN 1 \
                           ELSE 0 \
                           END as is_live_bet
                FROM bets
                WHERE bet_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                  AND status IN ('won', 'lost', 'pending') \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Chargement des paris ({days_back} derniers jours)...")
        return self.db.execute_query(query, {'days_back': days_back})

    def load_odds_history(self, days_back: int = 30, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge l'historique des cotes."""
        query = """
                SELECT outcome_id, \
                       odds_value, timestamp, change_type
                FROM odds_history
                WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY outcome_id, timestamp \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Chargement historique cotes ({days_back} jours)...")
        return self.db.execute_query(query, {'days_back': days_back})

    def load_all_data(self, config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données nécessaires."""
        config = config or {}

        logger.info("Chargement complet des données...")

        data = {
            'users': self.load_users_data(config.get('users_limit')),
            'events': self.load_events_data(
                config.get('events_days_back', 30),
                config.get('events_limit')
            ),
            'markets': self.load_markets_data(config.get('markets_limit')),
            'outcomes': self.load_outcomes_data(config.get('outcomes_limit')),
            'bets': self.load_bets_data(
                config.get('bets_days_back', 90),
                config.get('bets_limit')
            ),
            'odds_history': self.load_odds_history(
                config.get('odds_days_back', 30),
                config.get('odds_limit')
            )
        }

        for name, df in data.items():
            logger.info(f"{name}: {len(df)} enregistrements")

        return data

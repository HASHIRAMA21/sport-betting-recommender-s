from typing import Optional, Dict, Any

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
                SELECT id                                     AS user_id,
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
        # !-- WHERE status = 'active'

        if limit:
            query += f" LIMIT {limit}"

        logger.info("Chargement des données utilisateurs...")
        result = self.db.execute_query(query)
        logger.info(f"Utilisateurs chargés: {len(result)} enregistrements")
        return result

    def load_events_data(self, days_back: int = 2, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des événements sportifs."""
        query = """
                SELECT id                                   AS event_id,
                       event_type_name                      AS sport,
                       competition_name                     AS competition,
                       home_team,
                       away_team,
                       CONCAT(home_team, ' vs ', away_team) AS teams,
                       event_timestamp                      AS event_start_time,
                       event_status,
                       venue,
                       country_name                         AS country,
                       CASE
                           WHEN event_start_time > NOW() THEN TIMESTAMPDIFF(HOUR, NOW(), event_start_time)
                           ELSE 0
                           END                              AS hours_until_event
                FROM t_etl_event_summary
                WHERE event_start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
                  AND event_status IN ('scheduled', 'live', 'finished')
                """

        if limit:
            query += " LIMIT %s"
            params = (days_back, limit)
        else:
            params = (days_back,)

        logger.info(f"Chargement des événements ({days_back} derniers jours)...")
        result = self.db.execute_query(query, params)
        logger.info(f"Événements chargés: {len(result)} enregistrements")
        return result

    def load_bets_data(self, days_back: int = 365, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données de paris avec toutes les colonnes nécessaires."""

        # Version corrigée avec event_id inclus et jointure explicite
        query = """
                SELECT b.id      AS bet_id,
                       b.user_id,
                       b.event_id, -- AJOUTÉ : Colonne manquante !
                       b.outcome_id,
                       b.bet_amount,
                       b.odds    AS odds_used,
                       b.bet_timestamp,
                       b.settlement_timestamp,
                       b.outcome AS bet_status,
                       b.market,
                       b.bet_result,
                       CASE
                           WHEN b.bet_result = 'won' THEN 1
                           WHEN b.bet_result = 'lost' THEN 0
                           ELSE NULL
                           END   AS outcome,
                       CASE
                           WHEN e.event_start_time IS NOT NULL
                               AND TIMESTAMPDIFF(MINUTE, b.bet_timestamp, e.event_start_time) <= 0
                               THEN 1
                           ELSE 0
                           END   AS is_live_bet
                FROM t_etl_bet_summary b
                         LEFT JOIN t_etl_event_summary e ON b.event_id = e.id
                WHERE b.bet_timestamp >= '2020-01-01' -- Date fixe au lieu de DATE_SUB
                  AND b.bet_result IN ('won', 'lost', 'pending')
                """

        if limit:
            query += " LIMIT %s"
            params = (limit,)
        else:
            params = ()

        result = self.db.execute_query(query, params=params)
        logger.info(f"Paris chargés: {len(result)} lignes")

        if len(result) == 0:
            logger.warning("Aucun pari trouvé - exécution du diagnostic...")
            self._debug_bets_table(days_back)

        return result


    def load_markets_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des marchés de paris."""
        query = """
                SELECT market_id,
                       event_id,
                       market_name
                FROM t_etl_market_with_outcomes_summary
                """

        if limit:
            query += f" LIMIT {limit}"

        result = self.db.execute_query(query)
        logger.info(f"Marchés chargés: {len(result)} enregistrements")
        return result

    def load_outcomes_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des outcomes."""
        query = """
                SELECT outcome_id,
                       market_id,
                       outcome_name
                FROM t_etl_market_with_outcomes_summary
                """

        if limit:
            query += f" LIMIT {limit}"

        result = self.db.execute_query(query)
        logger.info(f"Outcomes chargés: {len(result)} enregistrements")
        return result

    def load_odds_history(self, days_back: int = 30, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge l'historique des cotes."""
        query = """
                SELECT outcome_id,
                       event_id,
                       odds AS odds_value, timestamp
                FROM t_etl_market_with_outcomes_summary
                ORDER BY outcome_id, timestamp
                """

        if limit:
            query += f" LIMIT {limit}"

        result = self.db.execute_query(query)
        logger.info(f"Historique cotes chargé: {len(result)} enregistrements")
        return result

    def diagnose_data_issues_1(self) -> Dict[str, Any]:
        """Diagnostique complet des problèmes de données."""

        diagnostics = {}

        # 1. Vérification de la table des paris
        logger.info("=== DIAGNOSTIC DES DONNÉES DE PARIS ===")

        bet_diagnostic_query = """
                               SELECT COUNT(*)                                                              as total_rows, \
                                      COUNT(CASE WHEN bet_amount IS NOT NULL AND bet_amount > 0 THEN 1 END) as valid_bet_amounts, \
                                      COUNT(CASE WHEN user_id IS NOT NULL THEN 1 END)                       as valid_user_ids, \
                                      COUNT(CASE WHEN event_id IS NOT NULL THEN 1 END)                      as valid_event_ids, \
                                      COUNT(CASE WHEN outcome_id IS NOT NULL THEN 1 END)                    as valid_outcome_ids, \
                                      MIN(bet_timestamp)                                                    as oldest_bet, \
                                      MAX(bet_timestamp)                                                    as newest_bet, \
                                      COUNT(DISTINCT user_id)                                               as unique_users, \
                                      COUNT(DISTINCT event_id)                                              as unique_events
                               FROM t_etl_bet_summary \
                               """

        try:
            bet_diag = self.db.execute_query(bet_diagnostic_query)
            diagnostics['bets'] = bet_diag.iloc[0].to_dict() if len(bet_diag) > 0 else {}
        except Exception as e:
            diagnostics['bets'] = {'error': str(e)}

        period_query = """
                       SELECT
                           DATE (bet_timestamp) as bet_date, COUNT(*) as daily_bets
                       FROM t_etl_bet_summary
                       WHERE bet_timestamp >= '2020-01-01'
                       GROUP BY DATE (bet_timestamp)
                       ORDER BY bet_date DESC
                           LIMIT 10 \
                       """

        try:
            period_diag = self.db.execute_query(period_query)
            diagnostics['recent_activity'] = period_diag.to_dict('records')
        except Exception as e:
            diagnostics['recent_activity'] = {'error': str(e)}

        # 3. Vérification des autres tables
        for table_name, query in [
            ('users', 'SELECT COUNT(*) as count FROM t_etl_user_summary'),
            ('events', 'SELECT COUNT(*) as count FROM t_etl_event_summary'),
            ('markets', 'SELECT COUNT(*) as count FROM t_etl_market_with_outcomes_summary')
        ]:
            try:
                result = self.db.execute_query(query)
                diagnostics[table_name] = result.iloc[0]['count'] if len(result) > 0 else 0
            except Exception as e:
                diagnostics[table_name] = f"Erreur: {str(e)}"

        # 4. Vérification des jointures (simplifiée)
        join_query = """
                     SELECT COUNT(*) as total_bets
                     FROM t_etl_bet_summary b
                     WHERE b.bet_timestamp >= '2020-01-01' \
                     """

        try:
            join_diag = self.db.execute_query(join_query)
            diagnostics['joins'] = join_diag.iloc[0].to_dict() if len(join_diag) > 0 else {}
        except Exception as e:
            diagnostics['joins'] = {'error': str(e)}

        # Affichage des résultats
        logger.info("=== RÉSULTATS DU DIAGNOSTIC ===")
        for key, value in diagnostics.items():
            logger.info(f"{key}: {value}")

        return diagnostics

    def _debug_bets_table(self, days_back: int):
        """Debug spécifique de la table des paris."""
        logger.info("=== DEBUG TABLE DES PARIS ===")

        # Vérification basique
        debug_queries = [
            ("Total des paris", "SELECT COUNT(*) as count FROM t_etl_bet_summary"),
            ("Paris avec montant",
             "SELECT COUNT(*) as count FROM t_etl_bet_summary WHERE bet_amount IS NOT NULL AND bet_amount > 0"),
            ("Paris récents",
             f"SELECT COUNT(*) as count FROM t_etl_bet_summary WHERE bet_timestamp >= DATE_SUB(NOW(), INTERVAL {days_back} DAY)"),
            ("Structure de la table", "DESCRIBE t_etl_bet_summary"),
            ("Échantillon de données", "SELECT * FROM t_etl_bet_summary ORDER BY bet_timestamp DESC LIMIT 5")
        ]

        for description, query in debug_queries:
            try:
                result = self.db.execute_query(query)
                logger.info(f"{description}: {result}")
            except Exception as e:
                logger.error(f"Erreur {description}: {e}")

    def load_all_data(self, config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données nécessaires avec diagnostics améliorés."""
        config = config or {}

        logger.info("=== CHARGEMENT COMPLET DES DONNÉES ===")

        # Diagnostic préliminaire
        if config.get('run_diagnostics', True):
            self.diagnose_data_issues_1()

        data = {}

        try:
            data['users'] = self.load_users_data(config.get('users_limit'))
        except Exception as e:
            logger.error(f"Erreur chargement utilisateurs: {e}")
            data['users'] = pd.DataFrame()

        try:
            data['events'] = self.load_events_data(
                config.get('events_days_back', 30),
                config.get('events_limit')
            )
        except Exception as e:
            logger.error(f"Erreur chargement événements: {e}")
            data['events'] = pd.DataFrame()

        try:
            data['markets'] = self.load_markets_data(config.get('markets_limit'))
        except Exception as e:
            logger.error(f"Erreur chargement marchés: {e}")
            data['markets'] = pd.DataFrame()

        try:
            data['outcomes'] = self.load_outcomes_data(config.get('outcomes_limit'))
        except Exception as e:
            logger.error(f"Erreur chargement outcomes: {e}")
            data['outcomes'] = pd.DataFrame()

        try:
            # Utilisation de la version flexible pour les paris
            data['bets'] = self.load_bets_data(
                config.get('bets_days_back', 365),  # Augmenté à 365 jours
                config.get('bets_limit')
            )
        except Exception as e:
            logger.error(f"Erreur chargement paris: {e}")
            data['bets'] = pd.DataFrame()

        try:
            data['odds_history'] = self.load_odds_history(
                config.get('odds_days_back', 30),
                config.get('odds_limit')
            )
        except Exception as e:
            logger.error(f"Erreur chargement historique cotes: {e}")
            data['odds_history'] = pd.DataFrame()

        # Résumé final
        logger.info("=== RÉSUMÉ DU CHARGEMENT ===")
        for name, df in data.items():
            logger.info(f"{name}: {len(df)} enregistrements")
            if len(df) == 0:
                logger.warning(f" {name}: AUCUNE DONNÉE CHARGÉE")


        return data

    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Génère un résumé détaillé des données chargées."""
        summary = {}

        for name, df in data.items():
            if len(df) > 0:
                summary[name] = {
                    'count': len(df),
                    'columns': list(df.columns),
                    'null_counts': df.isnull().sum().to_dict(),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }

                if name == 'bets' and len(df) > 0:
                    summary[name]['bet_amount_stats'] = df[
                        'bet_amount'].describe().to_dict() if 'bet_amount' in df.columns else 'N/A'
                    summary[name]['unique_users'] = df['user_id'].nunique() if 'user_id' in df.columns else 'N/A'
                    summary[name]['unique_events'] = df['event_id'].nunique() if 'event_id' in df.columns else 'N/A'

            else:
                summary[name] = {'count': 0, 'status': 'EMPTY'}

        return summary

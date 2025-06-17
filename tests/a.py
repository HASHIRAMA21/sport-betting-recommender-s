import os
import sys
import json
import pickle
import logging
import argparse
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

# Data & ML
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import entropy

# Database
import mysql.connector
from sqlalchemy import create_engine
import joblib

# MLflow
import mlflow
import mlflow.pytorch
import mlflow.catboost

# Disable warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlgorithmConfig:
    """Configuration unifiée pour tous les algorithmes."""

    # Général
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    n_jobs: int = -1

    # Collaborative Filtering
    cf_n_factors: int = 50
    cf_reg_lambda: float = 0.01
    cf_learning_rate: float = 0.01
    cf_epochs: int = 100
    cf_k_neighbors: int = 20

    # Content-Based Filtering
    cb_feature_dim: int = 128
    cb_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    cb_dropout: float = 0.3
    cb_epochs: int = 50
    cb_batch_size: int = 512

    # Hybrid Model
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        'collaborative': 0.4, 'content_based': 0.3, 'contextual': 0.3
    })
    hybrid_ensemble_method: str = 'weighted_average'

    # Contextual Bandits
    bandit_alpha: float = 1.0
    bandit_exploration_rate: float = 0.1
    bandit_update_frequency: int = 100
    bandit_context_dim: int = 32

    # Reinforcement Learning
    rl_state_dim: int = 64
    rl_action_dim: int = 32
    rl_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.95
    rl_epsilon: float = 0.1
    rl_epsilon_decay: float = 0.995
    rl_memory_size: int = 10000
    rl_batch_size: int = 32

    # CatBoost
    catboost_iterations: int = 1000
    catboost_learning_rate: float = 0.1
    catboost_depth: int = 6
    catboost_l2_leaf_reg: float = 3.0
    catboost_early_stopping_rounds: int = 50

    # Data Processing
    tfidf_max_features: int = 300
    rolling_windows: List[int] = field(default_factory=lambda: [7, 30, 90])
    handle_outliers: bool = True
    outlier_threshold: float = 1.5

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sports_betting_recommendation"

    # Performance
    enable_gpu: bool = True
    enable_mixed_precision: bool = True
    enable_parallel_training: bool = False  # Désactivé par défaut pour la stabilité

    # Database
    db_host: str = "146.59.148.113"
    db_port: int = 51724
    db_user: str = "sport_bet"
    db_password: str = "Sport@Bet19"
    db_name: str = "ai_engine_db"

    # Paths
    models_dir: str = "./models"
    artifacts_dir: str = "./artifacts"
    logs_dir: str = "./logs"


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_level: str = "INFO", log_dir: str = "./logs") -> logging.Logger:
    """Configure le système de logging."""

    # Créer le répertoire de logs
    Path(log_dir).mkdir(exist_ok=True)

    # Configuration du logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Logger principal
    logger = logging.getLogger("SportsRecommendation")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Handler pour fichier
    file_handler = logging.FileHandler(
        Path(log_dir) / f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    # Handler pour console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    # Ajouter les handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Logger global
logger = setup_logging()


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

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

            logger.info("✅ Connexion à la base de données établie")

        except Exception as e:
            logger.error(f"❌ Erreur connexion DB: {e}")
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


# =============================================================================
# DATA LOADER
# =============================================================================

class SportsDataLoader:
    """Chargeur de données spécialisé pour les paris sportifs."""

    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        self.data_cache = {}

    def load_users_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des utilisateurs."""
        query = """
                SELECT user_id, \
                       registration_date, \
                       country, \
                       age, \
                       total_deposits, \
                       total_withdrawals, \
                       vip_status, \
                       last_login_date, \
                       DATEDIFF(CURDATE(), registration_date) as days_since_registration
                FROM users
                WHERE status = 'active' \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info("Chargement des données utilisateurs...")
        return self.db.execute_query(query)

    def load_events_data(self, days_back: int = 30, limit: Optional[int] = None) -> pd.DataFrame:
        """Charge les données des événements sportifs."""
        query = """
                SELECT event_id, \
                       sport_type                           as sport, \
                       competition_name                     as competition, \
                       home_team, \
                       away_team, \
                       CONCAT(home_team, ' vs ', away_team) as teams, \
                       event_start_time, \
                       event_status, \
                       venue, \
                       country, \
                       CASE \
                           WHEN event_start_time > NOW() THEN TIMESTAMPDIFF(HOUR, NOW(), event_start_time) \
                           ELSE 0 \
                           END                              as hours_until_event
                FROM events
                WHERE event_start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
                  AND event_status IN ('scheduled', 'live', 'finished') \
                """

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Chargement des événements ({days_back} derniers jours)...")
        return self.db.execute_query(query, {'days_back': days_back})

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

        logger.info("🔄 Chargement complet des données...")

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

        # Statistiques de chargement
        for name, df in data.items():
            logger.info(f"  ✅ {name}: {len(df)} enregistrements")

        return data


# =============================================================================
# ADVANCED DATA PROCESSOR
# =============================================================================

class AdvancedDataProcessor:
    """Processeur de données avancé pour le système de recommandation."""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.data = {}
        self.processed_data = {}
        self.encoders = {}
        self.scalers = {}
        self.mappings = {}
        self.feature_names = {}

    def load_and_process_data(self, data_loader: SportsDataLoader) -> Dict[str, Any]:
        """Charge et traite toutes les données."""
        logger.info("🔄 Début du processing des données...")

        # Chargement des données brutes
        self.data = data_loader.load_all_data()

        # Vérification des données
        self._validate_data()

        # Processing étape par étape
        hierarchical_data = self._create_hierarchical_structure()
        user_features = self._create_user_features(hierarchical_data)
        event_features = self._create_event_features()
        interaction_matrix = self._create_interaction_matrix(hierarchical_data)
        training_data = self._prepare_training_data(hierarchical_data, user_features, event_features)

        self.processed_data = {
            'hierarchical_data': hierarchical_data,
            'user_features': user_features,
            'event_features': event_features,
            'interaction_matrix': interaction_matrix,
            'training_data': training_data
        }

        logger.info("✅ Processing des données terminé")
        return self.processed_data

    def _validate_data(self):
        """Valide les données chargées."""
        required_tables = ['users', 'events', 'markets', 'outcomes', 'bets']

        for table in required_tables:
            if table not in self.data or self.data[table].empty:
                raise ValueError(f"Table {table} manquante ou vide")

        logger.info("✅ Validation des données OK")

    def _create_hierarchical_structure(self) -> pd.DataFrame:
        """Crée la structure hiérarchique Event -> Market -> Outcome -> Bet."""
        logger.info("Création structure hiérarchique...")

        # Fusion progressive
        hierarchy = (
            self.data['events']
            .merge(self.data['markets'], on='event_id', how='inner', suffixes=('_event', '_market'))
            .merge(self.data['outcomes'], on='market_id', how='inner', suffixes=('', '_outcome'))
            .merge(self.data['bets'], on='outcome_id', how='left', suffixes=('', '_bet'))
            .merge(self.data['users'], on='user_id', how='left', suffixes=('', '_user'))
        )

        # Enrichissement temporel
        hierarchy = self._enrich_temporal_features(hierarchy)

        # Enrichissement avec historique des cotes
        if not self.data['odds_history'].empty:
            hierarchy = self._enrich_odds_features(hierarchy)

        logger.info(f"Structure hiérarchique: {len(hierarchy)} entrées")
        return hierarchy

    def _enrich_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des features temporelles."""
        df = df.copy()

        # Conversion des timestamps
        for col in ['event_start_time', 'bet_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Features temporelles pour les événements
        if 'event_start_time' in df.columns:
            now = pd.Timestamp.now()
            df['days_until_event'] = (df['event_start_time'] - now).dt.days
            df['event_month'] = df['event_start_time'].dt.month
            df['event_day_of_week'] = df['event_start_time'].dt.dayofweek
            df['event_hour'] = df['event_start_time'].dt.hour
            df['is_weekend_event'] = (df['event_day_of_week'] >= 5).astype(int)

            # Features cycliques
            df['event_hour_sin'] = np.sin(2 * np.pi * df['event_hour'] / 24)
            df['event_hour_cos'] = np.cos(2 * np.pi * df['event_hour'] / 24)

        # Features pour les paris
        if 'bet_timestamp' in df.columns:
            df['bet_hour'] = df['bet_timestamp'].dt.hour
            df['bet_day_of_week'] = df['bet_timestamp'].dt.dayofweek
            df['is_bet_weekend'] = (df['bet_day_of_week'] >= 5).astype(int)

        return df

    def _enrich_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit avec les statistiques des cotes."""
        odds_stats = self.data['odds_history'].groupby('outcome_id').agg({
            'odds_value': ['min', 'max', 'mean', 'std', 'count']
        }).reset_index()

        # Flatten columns
        odds_stats.columns = ['outcome_id', 'odds_min', 'odds_max', 'odds_mean', 'odds_std', 'odds_updates']

        # Features dérivées
        odds_stats['odds_volatility'] = odds_stats['odds_std'] / (odds_stats['odds_mean'] + 1e-8)
        odds_stats['odds_range'] = odds_stats['odds_max'] - odds_stats['odds_min']

        return df.merge(odds_stats, on='outcome_id', how='left')

    def _create_user_features(self, hierarchy: pd.DataFrame) -> pd.DataFrame:
        """Crée des features comportementales utilisateurs."""
        logger.info("Création features utilisateurs...")

        user_features = []

        for window in self.config.rolling_windows:
            # Filtrer par fenêtre temporelle
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=window)
            window_data = hierarchy[hierarchy['bet_timestamp'] >= cutoff_date]

            if window_data.empty:
                continue

            # Agrégations par utilisateur
            user_stats = window_data.groupby('user_id').agg({
                'bet_amount': ['count', 'sum', 'mean', 'std'],
                'odds_used': ['mean', 'std'],
                'outcome': ['mean', 'sum'],
                'sport': 'nunique',
                'market_name': 'nunique',
                'is_live_bet': 'mean'
            }).reset_index()

            # Renommage des colonnes
            new_cols = ['user_id']
            for metric in ['bet_count', 'total_stake', 'avg_stake', 'stake_std',
                           'avg_odds', 'odds_std', 'win_rate', 'total_wins',
                           'sports_diversity', 'markets_diversity', 'live_bet_rate']:
                new_cols.append(f"{metric}_{window}d")

            user_stats.columns = new_cols
            user_features.append(user_stats)

        # Merge des différentes fenêtres
        if user_features:
            result = user_features[0]
            for df in user_features[1:]:
                result = result.merge(df, on='user_id', how='outer')

            # Features dérivées
            self._add_derived_user_features(result)

            return result

        return pd.DataFrame()

    def _add_derived_user_features(self, df: pd.DataFrame):
        """Ajoute des features dérivées pour les utilisateurs."""
        # Tendances
        if 'win_rate_7d' in df.columns and 'win_rate_30d' in df.columns:
            df['win_rate_trend'] = df['win_rate_7d'] - df['win_rate_30d']

        if 'avg_stake_7d' in df.columns and 'avg_stake_30d' in df.columns:
            df['stake_trend'] = df['avg_stake_7d'] / (df['avg_stake_30d'] + 1e-8)

        # Segmentation RFM simplifiée
        if 'bet_count_30d' in df.columns:
            df['frequency_score'] = pd.qcut(df['bet_count_30d'].fillna(0), q=5, labels=range(1, 6), duplicates='drop')

        if 'total_stake_30d' in df.columns:
            df['monetary_score'] = pd.qcut(df['total_stake_30d'].fillna(0), q=5, labels=range(1, 6), duplicates='drop')

    def _create_event_features(self) -> np.ndarray:
        """Crée des features pour les événements."""
        logger.info("Création features événements...")

        events = self.data['events'].copy()

        # Encodage des variables catégorielles
        categorical_features = []
        for col in ['sport', 'competition', 'country']:
            if col in events.columns:
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(events[col].astype(str))
                categorical_features.append(encoded.reshape(-1, 1))
                self.encoders[f'event_{col}'] = encoder

        # Features numériques
        numeric_features = []
        for col in ['hours_until_event']:
            if col in events.columns:
                values = events[col].fillna(0).values.reshape(-1, 1)
                numeric_features.append(values)

        # Features temporelles
        if 'event_start_time' in events.columns:
            start_times = pd.to_datetime(events['event_start_time'])
            hour_features = start_times.dt.hour.values.reshape(-1, 1)
            dow_features = start_times.dt.dayofweek.values.reshape(-1, 1)
            numeric_features.extend([hour_features, dow_features])

        # Combinaison des features
        all_features = []

        if numeric_features:
            numeric_matrix = np.hstack(numeric_features)
            scaler = RobustScaler()
            numeric_matrix = scaler.fit_transform(numeric_matrix)
            all_features.append(numeric_matrix)
            self.scalers['event_numeric'] = scaler

        if categorical_features:
            categorical_matrix = np.hstack(categorical_features)
            all_features.append(categorical_matrix)

        if all_features:
            feature_matrix = np.hstack(all_features)
            self.mappings['event_ids'] = events['event_id'].values
            return feature_matrix

        return np.array([])

    def _create_interaction_matrix(self, hierarchy: pd.DataFrame) -> csr_matrix:
        """Crée la matrice d'interaction utilisateur-événement."""
        logger.info("Création matrice d'interaction...")

        # Filtrer les interactions valides
        interactions = hierarchy[
            hierarchy['bet_amount'].notna() &
            (hierarchy['bet_amount'] > 0)
            ].copy()

        if interactions.empty:
            return csr_matrix((0, 0))

        # Calcul des scores d'interaction
        user_event_scores = interactions.groupby(['user_id', 'event_id']).agg({
            'bet_amount': ['count', 'sum'],
            'outcome': ['mean', 'sum']
        }).reset_index()

        # Flatten columns
        user_event_scores.columns = ['user_id', 'event_id', 'frequency', 'total_amount', 'success_rate', 'wins']

        # Score composite
        scaler = MinMaxScaler()
        user_event_scores['frequency_norm'] = scaler.fit_transform(user_event_scores[['frequency']])
        user_event_scores['amount_norm'] = scaler.fit_transform(user_event_scores[['total_amount']])

        user_event_scores['interaction_score'] = (
                                                         0.4 * user_event_scores['frequency_norm'] +
                                                         0.3 * user_event_scores['amount_norm'] +
                                                         0.3 * user_event_scores['success_rate'].fillna(0)
                                                 ) * 5  # Échelle 0-5

        # Création des mappings
        unique_users = sorted(user_event_scores['user_id'].unique())
        unique_events = sorted(user_event_scores['event_id'].unique())

        user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        event_mapping = {eid: idx for idx, eid in enumerate(unique_events)}

        self.mappings['user_id_to_idx'] = user_mapping
        self.mappings['event_id_to_idx'] = event_mapping
        self.mappings['idx_to_user_id'] = {v: k for k, v in user_mapping.items()}
        self.mappings['idx_to_event_id'] = {v: k for k, v in event_mapping.items()}

        # Construction de la matrice sparse
        user_indices = [user_mapping[uid] for uid in user_event_scores['user_id']]
        event_indices = [event_mapping[eid] for eid in user_event_scores['event_id']]
        scores = user_event_scores['interaction_score'].values

        matrix = csr_matrix(
            (scores, (user_indices, event_indices)),
            shape=(len(unique_users), len(unique_events))
        )

        logger.info(f"Matrice: {matrix.shape}, sparsité: {1 - matrix.nnz / np.prod(matrix.shape):.4f}")
        return matrix

    def _prepare_training_data(self, hierarchy: pd.DataFrame, user_features: pd.DataFrame,
                               event_features: np.ndarray) -> Dict[str, Any]:
        """Prépare les données d'entraînement pour les modèles."""
        logger.info("Préparation données d'entraînement...")

        # Filtrer les données avec outcome connu
        training_samples = hierarchy[
            hierarchy['outcome'].notna() &
            hierarchy['bet_status'].isin(['won', 'lost'])
            ].copy()

        if training_samples.empty:
            logger.warning("Aucune donnée d'entraînement disponible")
            return {}

        # Merge avec user features
        if not user_features.empty:
            training_samples = training_samples.merge(user_features, on='user_id', how='left')

        # Sélection des features
        feature_columns = []

        # Features utilisateur
        user_cols = [col for col in training_samples.columns if
                     any(suffix in col for suffix in ['_7d', '_30d', '_90d'])]
        feature_columns.extend([col for col in user_cols if col in training_samples.columns])

        # Features de contexte
        context_cols = ['bet_amount', 'odds_used', 'is_live_bet', 'event_hour', 'event_day_of_week',
                        'is_weekend_event', 'days_until_event']
        feature_columns.extend([col for col in context_cols if col in training_samples.columns])

        # Features catégorielles
        categorical_cols = ['sport', 'market_name', 'country']
        for col in categorical_cols:
            if col in training_samples.columns:
                if col not in self.encoders:
                    encoder = LabelEncoder()
                    training_samples[col] = training_samples[col].astype(str).fillna('Unknown')
                    training_samples[col] = encoder.fit_transform(training_samples[col])
                    self.encoders[col] = encoder
                feature_columns.append(col)

        # Extraction des features et target
        available_features = [col for col in feature_columns if col in training_samples.columns]

        if not available_features:
            logger.warning("Aucune feature disponible pour l'entraînement")
            return {}

        X = training_samples[available_features].copy()
        y = training_samples['outcome'].copy()

        # Traitement des valeurs manquantes
        X = X.fillna(0)

        # Normalisation
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler

        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.config.val_size,
            random_state=self.config.random_state, stratify=y
        )

        logger.info(f"Données d'entraînement: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")

        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_names': available_features,
            'user_ids': training_samples['user_id'].values,
            'event_ids': training_samples['event_id'].values,
            'ratings': training_samples['outcome'].values
        }


# =============================================================================
# RECOMMENDATION MODELS
# =============================================================================

class BaseRecommender:
    """Classe de base pour tous les recommandeurs."""

    def __init__(self, config: AlgorithmConfig, name: str):
        self.config = config
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.training_time = 0

    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None):
        """Entraîne le modèle."""
        raise NotImplementedError

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores."""
        raise NotImplementedError

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items."""
        raise NotImplementedError

    def save_model(self, path: str):
        """Sauvegarde le modèle."""
        model_data = {
            'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else self.model,
            'config': self.config,
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'training_time': self.training_time
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Modèle {self.name} sauvegardé: {path}")


class MatrixFactorizationModel(nn.Module):
    """Modèle de factorisation matricielle avec PyTorch."""

    def __init__(self, n_users: int, n_items: int, n_factors: int, reg_lambda: float = 0.01):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda

        # Embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialisation
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot_product = (user_embedding * item_embedding).sum(dim=1)
        prediction = self.global_bias + user_bias + item_bias + dot_product

        return prediction

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(predictions, targets)

        # Régularisation L2
        reg_loss = (
                self.user_factors.weight.norm(2) +
                self.item_factors.weight.norm(2) +
                self.user_biases.weight.norm(2) +
                self.item_biases.weight.norm(2)
        )

        return mse_loss + self.reg_lambda * reg_loss


class CollaborativeFilteringRecommender(BaseRecommender):
    """Recommandeur par filtrage collaboratif."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "CollaborativeFiltering")
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None):
        """Entraîne le modèle collaboratif."""
        import time
        start_time = time.time()

        logger.info("🔄 Entraînement Collaborative Filtering...")

        # Extraction des données
        user_ids = train_data['user_ids']
        item_ids = train_data['event_ids']  # On utilise les événements comme items
        ratings = train_data['ratings'].astype(float)

        # Création des mappings
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)

        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Conversion en indices internes
        user_indices = np.array([self.user_mapping[uid] for uid in user_ids])
        item_indices = np.array([self.item_mapping[iid] for iid in item_ids])

        # Configuration du device
        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')
        logger.info(f"Utilisation du device: {device}")

        # Création du modèle
        self.model = MatrixFactorizationModel(
            n_users=n_users,
            n_items=n_items,
            n_factors=self.config.cf_n_factors,
            reg_lambda=self.config.cf_reg_lambda
        ).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.cf_learning_rate)

        # Conversion en tenseurs
        user_tensor = torch.LongTensor(user_indices).to(device)
        item_tensor = torch.LongTensor(item_indices).to(device)
        rating_tensor = torch.FloatTensor(ratings).to(device)

        # Entraînement
        self.model.train()
        best_loss = float('inf')
        patience = 0

        for epoch in range(self.config.cf_epochs):
            optimizer.zero_grad()

            predictions = self.model(user_tensor, item_tensor)
            loss = self.model.compute_loss(predictions, rating_tensor)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    logger.info(f"Early stopping à l'époque {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Époque {epoch}/{self.config.cf_epochs}, Loss: {current_loss:.4f}")

        # Extraction des facteurs
        self.user_factors = self.model.user_factors.weight.detach().cpu().numpy()
        self.item_factors = self.model.item_factors.weight.detach().cpu().numpy()
        self.user_biases = self.model.user_biases.weight.detach().cpu().numpy().flatten()
        self.item_biases = self.model.item_biases.weight.detach().cpu().numpy().flatten()
        self.global_bias = self.model.global_bias.item()

        self.is_trained = True
        self.training_time = time.time() - start_time

        # Calcul des métriques
        with torch.no_grad():
            train_predictions = self.model(user_tensor, item_tensor).cpu().numpy()
            train_mse = mean_squared_error(ratings, train_predictions)
            train_mae = mean_absolute_error(ratings, train_predictions)

            self.metrics = {
                'train_mse': train_mse,
                'train_mae': train_mae,
                'final_loss': best_loss
            }

        logger.info(f"✅ CF entraîné en {self.training_time:.2f}s - MSE: {train_mse:.4f}")
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores pour les paires user-item."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        predictions = []

        for uid, iid in zip(user_ids, item_ids):
            if uid in self.user_mapping and iid in self.item_mapping:
                user_idx = self.user_mapping[uid]
                item_idx = self.item_mapping[iid]

                pred = (
                        self.global_bias +
                        self.user_biases[user_idx] +
                        self.item_biases[item_idx] +
                        np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                )
                predictions.append(pred)
            else:
                predictions.append(self.global_bias)

        return np.array(predictions)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items."""
        if not self.is_trained or user_id not in self.user_mapping:
            return []

        user_idx = self.user_mapping[user_id]
        item_ids = list(self.item_mapping.keys())

        scores = []
        for item_id in item_ids:
            item_idx = self.item_mapping[item_id]
            score = (
                    self.global_bias +
                    self.user_biases[user_idx] +
                    self.item_biases[item_idx] +
                    np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            )
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]


class CatBoostRecommender(BaseRecommender):
    """Recommandeur utilisant CatBoost pour la prédiction contextuelle."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "CatBoostRecommender")
        self.feature_names = []
        self.categorical_features = []

    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None):
        """Entraîne le modèle CatBoost."""
        import time
        start_time = time.time()

        logger.info("🔄 Entraînement CatBoost...")

        X_train = train_data['X_train']
        y_train = train_data['y_train']
        self.feature_names = train_data.get('feature_names', [f'feature_{i}' for i in range(X_train.shape[1])])

        # Données de validation
        eval_set = None
        if val_data and 'X_val' in val_data:
            X_val = val_data['X_val']
            y_val = val_data['y_val']
            eval_set = (X_val, y_val)
        else:
            X_val = train_data.get('X_val')
            y_val = train_data.get('y_val')
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)

        # Configuration du modèle
        self.model = CatBoostClassifier(
            iterations=self.config.catboost_iterations,
            learning_rate=self.config.catboost_learning_rate,
            depth=self.config.catboost_depth,
            l2_leaf_reg=self.config.catboost_l2_leaf_reg,
            early_stopping_rounds=self.config.catboost_early_stopping_rounds,
            random_seed=self.config.random_state,
            verbose=50,
            task_type='GPU' if self.config.enable_gpu and torch.cuda.is_available() else 'CPU'
        )

        # Entraînement
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )

        self.is_trained = True
        self.training_time = time.time() - start_time

        # Métriques
        train_predictions = self.model.predict_proba(X_train)[:, 1]
        train_mse = mean_squared_error(y_train, train_predictions)

        self.metrics = {
            'train_mse': train_mse,
            'feature_importance': dict(zip(self.feature_names, self.model.get_feature_importance()))
        }

        logger.info(f"✅ CatBoost entraîné en {self.training_time:.2f}s - MSE: {train_mse:.4f}")
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit avec CatBoost."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")

        # Pour CatBoost, on a besoin des features complètes dans le contexte
        if context and 'features' in context:
            features = context['features']
            return self.model.predict_proba(features)[:, 1]
        else:
            # Fallback avec scores neutres
            return np.full(len(user_ids), 0.5)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommandations avec CatBoost."""
        if not self.is_trained:
            return []

        # Pour une vraie implémentation, il faudrait générer les features
        # pour tous les items candidats et prédire les scores
        return []


# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

class RecommendationTrainer:
    """Orchestrateur d'entraînement pour tous les modèles."""

    def __init__(self, config: AlgorithmConfig, data_processor: AdvancedDataProcessor):
        self.config = config
        self.data_processor = data_processor
        self.models = {}
        self.best_model = None

        # Configuration MLflow
        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment_name)
            self.mlflow_enabled = True
            logger.info("✅ MLflow configuré")
        except Exception as e:
            logger.warning(f"MLflow non disponible: {e}")
            self.mlflow_enabled = False

    def train_all_models(self, processed_data: Dict[str, Any]):
        """Entraîne tous les modèles disponibles."""
        logger.info("🚀 Début de l'entraînement des modèles...")

        training_data = processed_data['training_data']
        if not training_data:
            logger.error("Aucune donnée d'entraînement disponible")
            return

        # Liste des modèles à entraîner
        model_configs = [
            (CollaborativeFilteringRecommender, "collaborative_filtering"),
            (CatBoostRecommender, "catboost")
        ]

        for model_class, model_name in model_configs:
            logger.info(f"🔄 Entraînement {model_name}...")

            try:
                if self.mlflow_enabled:
                    with mlflow.start_run(run_name=f"{model_name}_training"):
                        model = self._train_single_model(model_class, training_data)
                        self._log_to_mlflow(model, model_name)
                else:
                    model = self._train_single_model(model_class, training_data)

                self.models[model_name] = model
                logger.info(f"✅ {model_name} entraîné avec succès")

            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Sélection du meilleur modèle
        self._select_best_model()

        logger.info(f"🎯 Entraînement terminé. Meilleur modèle: {self.best_model.name if self.best_model else 'Aucun'}")

    def _train_single_model(self, model_class, training_data: Dict[str, Any]):
        """Entraîne un modèle individuel."""
        model = model_class(self.config)

        # Séparation train/val pour ce modèle
        val_data = {
            'X_val': training_data.get('X_val'),
            'y_val': training_data.get('y_val')
        }

        model.fit(training_data, val_data)
        return model

    def _log_to_mlflow(self, model, model_name: str):
        """Log les métriques et le modèle dans MLflow."""
        try:
            # Log des paramètres
            mlflow.log_params({
                'model_name': model_name,
                'training_time': model.training_time,
                'random_state': self.config.random_state
            })

            # Log des métriques
            mlflow.log_metrics(model.metrics)

            # Sauvegarde du modèle
            model_path = f"{self.config.models_dir}/{model_name}_model.pkl"
            Path(self.config.models_dir).mkdir(exist_ok=True)
            model.save_model(model_path)
            mlflow.log_artifact(model_path)

        except Exception as e:
            logger.warning(f"Erreur MLflow logging: {e}")

    def _select_best_model(self):
        """Sélectionne le meilleur modèle basé sur les métriques."""
        if not self.models:
            return

        best_score = float('inf')
        best_name = None

        for name, model in self.models.items():
            # Utiliser MSE comme métrique principale (plus bas = meilleur)
            score = model.metrics.get('train_mse', float('inf'))

            if score < best_score:
                best_score = score
                best_name = name

        if best_name:
            self.best_model = self.models[best_name]
            logger.info(f"🏆 Meilleur modèle sélectionné: {best_name} (MSE: {best_score:.4f})")

    def get_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Obtient des recommandations du meilleur modèle."""
        if not self.best_model:
            logger.warning("Aucun modèle disponible pour les recommandations")
            return []

        try:
            return self.best_model.recommend(user_id, n_recommendations)
        except Exception as e:
            logger.error(f"Erreur lors des recommandations: {e}")
            return []


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class SportsRecommendationApp:
    """Application principale du système de recommandation."""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.db_connector = None
        self.data_loader = None
        self.data_processor = None
        self.trainer = None

        # Création des répertoires
        for path in [config.models_dir, config.artifacts_dir, config.logs_dir]:
            Path(path).mkdir(exist_ok=True)

    def initialize(self):
        """Initialise tous les composants."""
        logger.info("🔧 Initialisation de l'application...")

        # Connexion à la base de données
        self.db_connector = DatabaseConnector(self.config)
        self.db_connector.connect()

        # Chargeur de données
        self.data_loader = SportsDataLoader(self.db_connector)

        # Processeur de données
        self.data_processor = AdvancedDataProcessor(self.config)

        logger.info("✅ Initialisation terminée")

    def run_training_pipeline(self, data_config: Optional[Dict] = None):
        """Exécute le pipeline complet d'entraînement."""
        logger.info("🚀 Démarrage du pipeline d'entraînement...")

        try:
            # 1. Chargement et processing des données
            logger.info("📊 Phase 1: Chargement et traitement des données")
            processed_data = self.data_processor.load_and_process_data(self.data_loader)

            if not processed_data or not processed_data.get('training_data'):
                logger.error("❌ Échec du processing des données")
                return False

            # 2. Entraînement des modèles
            logger.info("🤖 Phase 2: Entraînement des modèles")
            self.trainer = RecommendationTrainer(self.config, self.data_processor)
            self.trainer.train_all_models(processed_data)

            if not self.trainer.models:
                logger.error("❌ Aucun modèle entraîné avec succès")
                return False

            # 3. Sauvegarde des artefacts
            logger.info("💾 Phase 3: Sauvegarde des artefacts")
            self._save_artifacts()

            logger.info("🎉 Pipeline d'entraînement terminé avec succès!")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur dans le pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            if self.db_connector:
                self.db_connector.disconnect()

    def _save_artifacts(self):
        """Sauvegarde tous les artefacts."""
        artifacts_path = Path(self.config.artifacts_dir)

        # Sauvegarde des encoders et scalers
        processor_artifacts = {
            'encoders': self.data_processor.encoders,
            'scalers': self.data_processor.scalers,
            'mappings': self.data_processor.mappings,
            'feature_names': self.data_processor.feature_names,
            'config': self.config
        }

        with open(artifacts_path / 'processor_artifacts.pkl', 'wb') as f:
            pickle.dump(processor_artifacts, f)

        # Sauvegarde des modèles
        for name, model in self.trainer.models.items():
            model_path = artifacts_path / f"{name}_model.pkl"
            model.save_model(str(model_path))

        # Sauvegarde de la configuration
        config_dict = {
            key: getattr(self.config, key)
            for key in dir(self.config)
            if not key.startswith('_')
        }

        with open(artifacts_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"💾 Artefacts sauvegardés dans {artifacts_path}")

    def get_recommendations_for_user(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Interface pour obtenir des recommandations."""
        if not self.trainer or not self.trainer.best_model:
            logger.error("Aucun modèle disponible. Veuillez d'abord entraîner les modèles.")
            return []

        return self.trainer.get_recommendations(user_id, n_recommendations)

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur les modèles entraînés."""
        if not self.trainer:
            return {}

        info = {}
        for name, model in self.trainer.models.items():
            info[name] = {
                'is_trained': model.is_trained,
                'training_time': model.training_time,
                'metrics': model.metrics
            }

        if self.trainer.best_model:
            info['best_model'] = self.trainer.best_model.name

        return info


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Système de Recommandation de Paris Sportifs")

    # Actions
    parser.add_argument('action', choices=['train', 'recommend', 'info'],
                        help='Action à effectuer')

    # Paramètres généraux
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Niveau de logging')

    # Paramètres de base de données
    parser.add_argument('--db-host', default='146.59.148.113', help='Hôte de la base de données')
    parser.add_argument('--db-port', type=int, default=51724, help='Port de la base de données')
    parser.add_argument('--db-user', default='sport_bet', help='Utilisateur de la base de données')
    parser.add_argument('--db-password', default='Sport@Bet19', help='Mot de passe de la base de données')
    parser.add_argument('--db-name', default='ai_engine_db', help='Nom de la base de données')

    # Paramètres d'entraînement
    parser.add_argument('--cf-factors', type=int, default=50, help='Nombre de facteurs pour CF')
    parser.add_argument('--cf-epochs', type=int, default=100, help='Nombre d\'époques pour CF')
    parser.add_argument('--catboost-iterations', type=int, default=1000, help='Itérations CatBoost')

    # Paramètres de recommandation
    parser.add_argument('--user-id', type=int, help='ID utilisateur pour recommandations')
    parser.add_argument('--n-recommendations', type=int, default=10, help='Nombre de recommandations')

    # Répertoires
    parser.add_argument('--models-dir', default='./models', help='Répertoire des modèles')
    parser.add_argument('--artifacts-dir', default='./artifacts', help='Répertoire des artefacts')
    parser.add_argument('--logs-dir', default='./logs', help='Répertoire des logs')

    # Performance
    parser.add_argument('--enable-gpu', action='store_true', help='Activer le GPU si disponible')
    parser.add_argument('--disable-mlflow', action='store_true', help='Désactiver MLflow')

    return parser.parse_args()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis un fichier JSON."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return {}


def create_config_from_args(args) -> AlgorithmConfig:
    """Crée un objet AlgorithmConfig à partir des arguments."""
    config = AlgorithmConfig()

    # Si un fichier de config est spécifié, le charger d'abord
    if args.config:
        file_config = load_config_from_file(args.config)
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Override avec les arguments de ligne de commande
    if args.db_host:
        config.db_host = args.db_host
    if args.db_port:
        config.db_port = args.db_port
    if args.db_user:
        config.db_user = args.db_user
    if args.db_password:
        config.db_password = args.db_password
    if args.db_name:
        config.db_name = args.db_name

    if args.cf_factors:
        config.cf_n_factors = args.cf_factors
    if args.cf_epochs:
        config.cf_epochs = args.cf_epochs
    if args.catboost_iterations:
        config.catboost_iterations = args.catboost_iterations

    if args.models_dir:
        config.models_dir = args.models_dir
    if args.artifacts_dir:
        config.artifacts_dir = args.artifacts_dir
    if args.logs_dir:
        config.logs_dir = args.logs_dir

    if args.enable_gpu:
        config.enable_gpu = True

    return config


def main():
    """Fonction principale de l'application."""
    # Parse des arguments
    args = parse_arguments()

    # Configuration du logging
    global logger
    logger = setup_logging(args.log_level, args.logs_dir)

    logger.info("🎯 Démarrage du Système de Recommandation de Paris Sportifs")
    logger.info(f"Action demandée: {args.action}")

    try:
        # Création de la configuration
        config = create_config_from_args(args)

        # Initialisation de l'application
        app = SportsRecommendationApp(config)
        app.initialize()

        # Exécution de l'action demandée
        if args.action == 'train':
            logger.info("🚀 Lancement de l'entraînement...")
            success = app.run_training_pipeline()

            if success:
                logger.info("✅ Entraînement terminé avec succès!")

                # Affichage des informations sur les modèles
                model_info = app.get_model_info()
                logger.info("📊 Informations sur les modèles:")
                for name, info in model_info.items():
                    if name != 'best_model':
                        logger.info(f"  {name}:")
                        logger.info(f"    - Entraîné: {info['is_trained']}")
                        logger.info(f"    - Temps: {info['training_time']:.2f}s")
                        logger.info(f"    - MSE: {info['metrics'].get('train_mse', 'N/A')}")

                if 'best_model' in model_info:
                    logger.info(f"🏆 Meilleur modèle: {model_info['best_model']}")
            else:
                logger.error("❌ Échec de l'entraînement")
                return 1

        elif args.action == 'recommend':
            if not args.user_id:
                logger.error("❌ --user-id requis pour les recommandations")
                return 1

            logger.info(f"🎯 Génération de recommandations pour l'utilisateur {args.user_id}")

            # Chargement des modèles existants si nécessaire
            if not hasattr(app, 'trainer') or not app.trainer:
                logger.info("📥 Chargement des modèles existants...")
                app._load_existing_models()

            recommendations = app.get_recommendations_for_user(
                args.user_id,
                args.n_recommendations
            )

            if recommendations:
                logger.info(f"✅ {len(recommendations)} recommandations générées:")
                for i, (item_id, score) in enumerate(recommendations, 1):
                    logger.info(f"  {i}. Item {item_id}: {score:.4f}")
            else:
                logger.warning("⚠️ Aucune recommandation générée")

        elif args.action == 'info':
            logger.info("📊 Informations sur le système:")

            # Informations sur la base de données
            try:
                users_count = app.data_loader.db.execute_query("SELECT COUNT(*) as count FROM users").iloc[0]['count']
                events_count = app.data_loader.db.execute_query("SELECT COUNT(*) as count FROM events").iloc[0]['count']
                bets_count = app.data_loader.db.execute_query("SELECT COUNT(*) as count FROM bets").iloc[0]['count']

                logger.info(f"  Database:")
                logger.info(f"    - Utilisateurs: {users_count:,}")
                logger.info(f"    - Événements: {events_count:,}")
                logger.info(f"    - Paris: {bets_count:,}")

            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des stats DB: {e}")

            # Informations sur les modèles
            model_info = app.get_model_info()
            if model_info:
                logger.info("  Modèles disponibles:")
                for name, info in model_info.items():
                    if name != 'best_model':
                        logger.info(f"    - {name}: {'✅' if info['is_trained'] else '❌'}")
            else:
                logger.info("  Aucun modèle entraîné trouvé")

        logger.info("🎉 Exécution terminée avec succès!")
        return 0

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


# Extension de la classe SportsRecommendationApp pour le chargement de modèles
def _load_existing_models(self):
    """Charge les modèles existants depuis les artefacts."""
    artifacts_path = Path(self.config.artifacts_dir)

    if not artifacts_path.exists():
        logger.warning("Aucun répertoire d'artefacts trouvé")
        return

    try:
        # Chargement des artefacts du processeur
        processor_artifacts_path = artifacts_path / 'processor_artifacts.pkl'
        if processor_artifacts_path.exists():
            with open(processor_artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)

            self.data_processor = AdvancedDataProcessor(self.config)
            self.data_processor.encoders = artifacts['encoders']
            self.data_processor.scalers = artifacts['scalers']
            self.data_processor.mappings = artifacts['mappings']
            self.data_processor.feature_names = artifacts['feature_names']

            logger.info("✅ Artefacts du processeur chargés")

        # Chargement des modèles
        self.trainer = RecommendationTrainer(self.config, self.data_processor)
        self.trainer.models = {}

        for model_file in artifacts_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')

            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                # Reconstruction du modèle (simplifié)
                if model_name == 'collaborative_filtering':
                    model = CollaborativeFilteringRecommender(self.config)
                elif model_name == 'catboost':
                    model = CatBoostRecommender(self.config)
                else:
                    continue

                # Restauration des attributs
                for attr in ['is_trained', 'metrics', 'training_time']:
                    if attr in model_data:
                        setattr(model, attr, model_data[attr])

                self.trainer.models[model_name] = model
                logger.info(f"✅ Modèle {model_name} chargé")

            except Exception as e:
                logger.warning(f"⚠️ Erreur lors du chargement de {model_name}: {e}")

        # Sélection du meilleur modèle
        self.trainer._select_best_model()

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")


# Ajout de la méthode à la classe
SportsRecommendationApp._load_existing_models = _load_existing_models


# =============================================================================
# EXEMPLE D'UTILISATION AVEC DONNÉES SIMULÉES
# =============================================================================

def create_sample_config() -> AlgorithmConfig:
    """Crée une configuration d'exemple pour les tests."""
    config = AlgorithmConfig()

    # Configuration pour tests locaux
    config.cf_epochs = 50  # Réduction pour tests plus rapides
    config.catboost_iterations = 100
    config.enable_gpu = False  # Désactivation GPU pour compatibilité
    config.mlflow_tracking_uri = "./mlruns"  # MLflow local

    return config


def run_example_with_simulated_data():
    """Exemple d'utilisation avec des données simulées."""
    logger.info("🧪 Exécution avec données simulées...")

    # Configuration
    config = create_sample_config()

    # Données simulées pour test
    np.random.seed(config.random_state)

    n_users = 1000
    n_events = 200
    n_bets = 5000

    # Simulation des données
    simulated_data = {
        'users': pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'registration_date': pd.date_range('2020-01-01', periods=n_users, freq='1D'),
            'country': np.random.choice(['FR', 'UK', 'DE', 'ES'], n_users),
            'age': np.random.randint(18, 65, n_users),
            'total_deposits': np.random.lognormal(6, 1, n_users),
            'total_withdrawals': np.random.lognormal(5, 1, n_users),
            'vip_status': np.random.choice([0, 1], n_users, p=[0.9, 0.1]),
            'last_login_date': pd.date_range('2024-01-01', periods=n_users, freq='1H'),
            'days_since_registration': np.random.randint(30, 1000, n_users)
        }),

        'events': pd.DataFrame({
            'event_id': range(1, n_events + 1),
            'sport': np.random.choice(['Football', 'Basketball', 'Tennis', 'Hockey'], n_events),
            'competition': np.random.choice(['League1', 'League2', 'Cup'], n_events),
            'home_team': [f'Team_{i}_Home' for i in range(n_events)],
            'away_team': [f'Team_{i}_Away' for i in range(n_events)],
            'teams': [f'Team_{i}_Home vs Team_{i}_Away' for i in range(n_events)],
            'event_start_time': pd.date_range('2024-01-01', periods=n_events, freq='6H'),
            'event_status': np.random.choice(['scheduled', 'live', 'finished'], n_events),
            'venue': [f'Stadium_{i}' for i in range(n_events)],
            'country': np.random.choice(['FR', 'UK', 'DE', 'ES'], n_events),
            'hours_until_event': np.random.randint(-24, 168, n_events)
        }),

        'markets': pd.DataFrame({
            'market_id': range(1, n_events * 3 + 1),
            'event_id': np.repeat(range(1, n_events + 1), 3),
            'market_name': np.tile(['Match Winner', 'Over/Under 2.5', 'Both Teams Score'], n_events),
            'market_type': np.tile(['win_draw_win', 'over_under', 'both_score'], n_events),
            'market_status': np.random.choice(['open', 'suspended', 'closed'], n_events * 3),
            'market_open_time': pd.date_range('2024-01-01', periods=n_events * 3, freq='2H')
        }),

        'outcomes': pd.DataFrame({
            'outcome_id': range(1, n_events * 3 * 3 + 1),
            'market_id': np.repeat(range(1, n_events * 3 + 1), 3),
            'outcome_name': np.tile(['Option1', 'Option2', 'Option3'], n_events * 3),
            'current_odds': np.random.uniform(1.2, 5.0, n_events * 3 * 3),
            'outcome_status': np.random.choice(['active', 'suspended', 'settled'], n_events * 3 * 3)
        }),

        'bets': pd.DataFrame({
            'bet_id': range(1, n_bets + 1),
            'user_id': np.random.randint(1, n_users + 1, n_bets),
            'outcome_id': np.random.randint(1, n_events * 3 * 3 + 1, n_bets),
            'bet_amount': np.random.lognormal(2, 1, n_bets),
            'odds_used': np.random.uniform(1.2, 5.0, n_bets),
            'bet_timestamp': pd.date_range('2024-01-01', periods=n_bets, freq='1H'),
            'settlement_timestamp': pd.date_range('2024-01-02', periods=n_bets, freq='1H'),
            'bet_status': np.random.choice(['won', 'lost', 'pending'], n_bets, p=[0.4, 0.5, 0.1]),
            'outcome': np.random.choice([0, 1], n_bets, p=[0.6, 0.4]),
            'is_live_bet': np.random.choice([0, 1], n_bets, p=[0.7, 0.3])
        }),

        'odds_history': pd.DataFrame({
            'outcome_id': np.random.randint(1, n_events * 3 * 3 + 1, 1000),
            'odds_value': np.random.uniform(1.2, 5.0, 1000),
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1H'),
            'change_type': np.random.choice(['increase', 'decrease'], 1000)
        })
    }

    # Simulation d'un data loader
    class SimulatedDataLoader:
        def __init__(self, data):
            self.data = data

        def load_all_data(self):
            return self.data

    # Traitement avec le processeur
    processor = AdvancedDataProcessor(config)
    processor.data = simulated_data

    try:
        # Processing des données
        logger.info("🔄 Processing des données simulées...")
        hierarchical_data = processor._create_hierarchical_structure()
        user_features = processor._create_user_features(hierarchical_data)
        event_features = processor._create_event_features()
        interaction_matrix = processor._create_interaction_matrix(hierarchical_data)
        training_data = processor._prepare_training_data(hierarchical_data, user_features, event_features)

        processed_data = {
            'hierarchical_data': hierarchical_data,
            'user_features': user_features,
            'event_features': event_features,
            'interaction_matrix': interaction_matrix,
            'training_data': training_data
        }

        # Entraînement
        if training_data:
            logger.info("🤖 Entraînement des modèles...")
            trainer = RecommendationTrainer(config, processor)
            trainer.train_all_models(processed_data)

            # Test des recommandations
            if trainer.models:
                logger.info("🎯 Test des recommandations...")
                recommendations = trainer.get_recommendations(123, 5)
                logger.info(f"Recommandations pour utilisateur 123: {recommendations}")

            logger.info("✅ Test avec données simulées terminé!")
        else:
            logger.warning("⚠️ Aucune donnée d'entraînement générée")

    except Exception as e:
        logger.error(f"❌ Erreur dans le test simulé: {e}")
        import traceback
        logger.error(traceback.format_exc())


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Vérification des arguments
    if len(sys.argv) == 1:
        # Mode test avec données simulées si aucun argument
        logger.info("🧪 Mode test avec données simulées")
        run_example_with_simulated_data()
    else:
        # Mode normal avec arguments CLI
        exit_code = main()
        sys.exit(exit_code)
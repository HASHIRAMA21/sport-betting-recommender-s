import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from feast import Entity, Feature, FeatureView, ValueType, FileSource
from scipy.sparse import csr_matrix, hstack
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler

from configs.processing_config import ProcessingConfig
from helper.logging import logger


class DataProcessor:
    """
    Processeur de données avancé pour systèmes de recommandation de paris sportifs.

    Gère la structure hiérarchique Event -> Market -> Outcome -> Odds
    Compatible avec les fournisseurs comme Sportradar, BetRadar, etc.
    Intégration native avec Feast Feature Store.
    """

    def __init__(self,
                 df_users: pd.DataFrame,
                 df_events: pd.DataFrame,
                 df_markets: pd.DataFrame,
                 df_outcomes: pd.DataFrame,
                 df_bets: pd.DataFrame,
                 df_odds_history: Optional[pd.DataFrame] = None,
                 config: Optional[ProcessingConfig] = None):
        """
        Initialise le processeur avec la structure Sportradar complète.

        Args:
            df_users: DataFrame des utilisateurs
            df_events: DataFrame des événements sportifs
            df_markets: DataFrame des marchés de paris
            df_outcomes: DataFrame des résultats possibles
            df_bets: DataFrame des paris placés
            df_odds_history: DataFrame historique des cotes (optionnel)
            config: Configuration de traitement
        """
        self.config = config or ProcessingConfig()

        # Validation et stockage des DataFrames
        self.df_users = self._validate_and_copy_dataframe(df_users, "users")
        self.df_events = self._validate_and_copy_dataframe(df_events, "events")
        self.df_markets = self._validate_and_copy_dataframe(df_markets, "markets")
        self.df_outcomes = self._validate_and_copy_dataframe(df_outcomes, "outcomes")
        self.df_bets = self._validate_and_copy_dataframe(df_bets, "bets")
        self.df_odds_history = df_odds_history.copy() if df_odds_history is not None else None

        # Structures de données traitées
        self.merged_data = None
        self.hierarchical_data = None
        self.user_item_matrix = None
        self.event_features = None
        self.market_features = None
        self.user_features = None
        self.temporal_features = None
        self.odds_features = None

        # Composants ML
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        # Mappings et métadonnées
        self.user_id_mapping = {}
        self.event_id_mapping = {}
        self.market_id_mapping = {}
        self.outcome_id_mapping = {}
        self.feature_names = {}
        self.feature_importance = {}

        # Feature Store
        self.feast_store = None
        if self.config.use_feast:
            self._init_feast_store()

        logger.info(f"Processeur initialisé avec {len(self.df_users)} utilisateurs, "
                   f"{len(self.df_events)} événements, {len(self.df_markets)} marchés")

    def _validate_and_copy_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Valide et copie un DataFrame avec vérifications spécifiques."""
        if df is None or df.empty:
            raise ValueError(f"DataFrame {name} est vide ou None")

        # Vérifications spécifiques par type
        required_columns = {
            'users': ['user_id'],
            'events': ['event_id', 'sport'],
            'markets': ['market_id', 'event_id', 'market_name'],
            'outcomes': ['outcome_id', 'market_id', 'outcome_name'],
            'bets': ['bet_id', 'user_id', 'outcome_id', 'bet_amount', 'odds_used']
        }

        if name in required_columns:
            missing_cols = set(required_columns[name]) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans {name}: {missing_cols}")

        logger.info(f"Validation {name}: {len(df)} lignes, {len(df.columns)} colonnes")
        return df.copy()

    def _init_feast_store(self):
        """Initialise le Feature Store Feast."""
        try:
            from feast import FeatureStore
            self.feast_store = FeatureStore(repo_path=self.config.feast_repo_path)
            logger.info("Feature Store Feast initialisé")
        except ImportError:
            logger.warning("Feast non disponible. Feature Store désactivé.")
            self.config.use_feast = False
        except Exception as e:
            logger.error(f"Erreur initialisation Feast: {e}")
            self.config.use_feast = False

    def create_hierarchical_data(self) -> pd.DataFrame:
        """
        Crée la structure hiérarchique Event -> Market -> Outcome -> Odds avec enrichissement.
        """
        logger.info("Création de la structure hiérarchique des données...")

        try:
            # Construction hiérarchique progressive
            # 1. Events + Markets
            events_markets = self.df_events.merge(
                self.df_markets, on='event_id', how='inner', suffixes=('_event', '_market')
            )

            # 2. + Outcomes
            full_hierarchy = events_markets.merge(
                self.df_outcomes, on='market_id', how='inner', suffixes=('', '_outcome')
            )

            # 3. + Bets (pour les interactions)
            hierarchy_with_bets = full_hierarchy.merge(
                self.df_bets, on='outcome_id', how='left', suffixes=('', '_bet')
            )

            # 4. + Users (profil)
            complete_hierarchy = hierarchy_with_bets.merge(
                self.df_users, on='user_id', how='left', suffixes=('', '_user')
            )

            # Enrichissement temporel
            self._enrich_temporal_data(complete_hierarchy)

            # Enrichissement des cotes
            if self.df_odds_history is not None:
                self._enrich_odds_data(complete_hierarchy)

            # Features dérivées pour la hiérarchie
            self._create_hierarchy_derived_features(complete_hierarchy)

            self.hierarchical_data = complete_hierarchy
            logger.info(f"Structure hiérarchique créée: {len(complete_hierarchy)} entrées")

            return complete_hierarchy

        except Exception as e:
            logger.error(f"Erreur création hiérarchie: {e}")
            raise

    def _enrich_temporal_data(self, df: pd.DataFrame):
        """Enrichit les données avec des features temporelles avancées."""

        # Conversion des timestamps
        timestamp_columns = ['event_start_time', 'bet_timestamp', 'market_open_time']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Features temporelles pour events
        if 'event_start_time' in df.columns:
            now = pd.Timestamp.now()
            df['days_until_event'] = (df['event_start_time'] - now).dt.days
            df['hours_until_event'] = (df['event_start_time'] - now).dt.total_seconds() / 3600
            df['event_month'] = df['event_start_time'].dt.month
            df['event_day_of_week'] = df['event_start_time'].dt.dayofweek
            df['event_hour'] = df['event_start_time'].dt.hour
            df['is_weekend_event'] = (df['event_day_of_week'] >= 5).astype(int)

            # Features cycliques
            df['event_hour_sin'] = np.sin(2 * np.pi * df['event_hour'] / 24)
            df['event_hour_cos'] = np.cos(2 * np.pi * df['event_hour'] / 24)
            df['event_month_sin'] = np.sin(2 * np.pi * df['event_month'] / 12)
            df['event_month_cos'] = np.cos(2 * np.pi * df['event_month'] / 12)

        # Features temporelles pour bets
        if 'bet_timestamp' in df.columns:
            df['bet_hour'] = df['bet_timestamp'].dt.hour
            df['bet_day_of_week'] = df['bet_timestamp'].dt.dayofweek
            df['is_bet_weekend'] = (df['bet_day_of_week'] >= 5).astype(int)

            # Timing par rapport à l'événement
            if 'event_start_time' in df.columns:
                df['bet_timing_hours_before'] = (df['event_start_time'] - df['bet_timestamp']).dt.total_seconds() / 3600
                df['is_live_bet'] = (df['bet_timing_hours_before'] <= 0).astype(int)

    def _enrich_odds_data(self, df: pd.DataFrame):
        """Enrichit avec l'historique des cotes."""
        if self.df_odds_history is None:
            return

        # Calcul des mouvements de cotes
        odds_stats = self.df_odds_history.groupby('outcome_id').agg({
            'odds_value': ['min', 'max', 'mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).reset_index()

        # Flatten columns
        odds_stats.columns = ['outcome_id', 'odds_min', 'odds_max', 'odds_mean',
                             'odds_std', 'odds_updates_count', 'first_odds_time', 'last_odds_time']

        # Features dérivées
        odds_stats['odds_volatility'] = odds_stats['odds_std'] / odds_stats['odds_mean']
        odds_stats['odds_range'] = odds_stats['odds_max'] - odds_stats['odds_min']
        odds_stats['odds_drift'] = (odds_stats['odds_max'] - odds_stats['odds_min']) / odds_stats['odds_min']

        # Merge avec les données principales
        df = df.merge(odds_stats, on='outcome_id', how='left')

    def _create_hierarchy_derived_features(self, df: pd.DataFrame):
        """Crée des features dérivées de la structure hiérarchique."""

        # Features au niveau Event
        event_stats = df.groupby('event_id').agg({
            'market_id': 'nunique',  # Nombre de marchés
            'outcome_id': 'nunique',  # Nombre d'outcomes
            'bet_amount': ['count', 'sum', 'mean'],  # Stats des paris
            'odds_used': ['mean', 'std']
        }).reset_index()

        event_stats.columns = ['event_id', 'num_markets', 'num_outcomes',
                              'total_bets', 'total_volume', 'avg_bet_amount',
                              'avg_odds', 'odds_std']

        event_stats['event_popularity'] = event_stats['total_bets']
        event_stats['event_liquidity'] = event_stats['total_volume']

        # Features au niveau Market
        market_stats = df.groupby('market_id').agg({
            'outcome_id': 'nunique',
            'bet_amount': ['count', 'sum', 'mean'],
            'odds_used': ['mean', 'std']
        }).reset_index()

        market_stats.columns = ['market_id', 'market_num_outcomes', 'market_bet_count',
                               'market_volume', 'market_avg_bet', 'market_avg_odds', 'market_odds_std']

        # Merge back
        df = df.merge(event_stats, on='event_id', how='left')
        df = df.merge(market_stats, on='market_id', how='left')

    def create_user_behavioral_features(self) -> pd.DataFrame:
        """Crée des features comportementales sophistiquées pour les utilisateurs."""
        logger.info("Création des features comportementales utilisateurs...")

        if self.hierarchical_data is None:
            self.create_hierarchical_data()

        df = self.hierarchical_data.copy()

        # Features par utilisateur avec fenêtres glissantes
        user_features = {}

        for window in self.config.rolling_windows:
            window_data = df[df['bet_timestamp'] >= (pd.Timestamp.now() - pd.Timedelta(days=window))]

            user_window_stats = window_data.groupby('user_id').agg({
                'bet_amount': ['count', 'sum', 'mean', 'std'],
                'odds_used': ['mean', 'std', 'min', 'max'],
                'outcome': ['sum', lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0],
                'sport': ['nunique'],
                'market_name': ['nunique'],
                'is_live_bet': 'mean'
            }).reset_index()

            # Flatten et renommer
            cols = ['user_id']
            for stat_type in ['bet_count', 'total_stake', 'avg_stake', 'stake_std',
                             'avg_odds', 'odds_std', 'min_odds', 'max_odds',
                             'total_wins', 'win_rate', 'sports_diversity', 'market_diversity', 'live_bet_rate']:
                cols.append(f"{stat_type}_{window}d")

            user_window_stats.columns = cols
            user_features[f'{window}d'] = user_window_stats

        # Merge toutes les fenêtres
        user_behavioral = user_features['7d']  # Base
        for window in ['14d', '30d', '90d']:
            if window in user_features:
                user_behavioral = user_behavioral.merge(
                    user_features[window], on='user_id', how='outer', suffixes=('', f'_{window}')
                )

        # Features dérivées
        if 'win_rate_7d' in user_behavioral.columns and 'win_rate_30d' in user_behavioral.columns:
            user_behavioral['win_rate_trend'] = user_behavioral['win_rate_7d'] - user_behavioral['win_rate_30d']

        if 'avg_stake_7d' in user_behavioral.columns and 'avg_stake_30d' in user_behavioral.columns:
            user_behavioral['stake_trend'] = user_behavioral['avg_stake_7d'] / (user_behavioral['avg_stake_30d'] + 1)

        # Entropy pour diversité sportive
        sports_entropy = df.groupby('user_id')['sport'].apply(
            lambda x: entropy(x.value_counts(normalize=True)) if len(x) > 1 else 0
        ).reset_index()
        sports_entropy.columns = ['user_id', 'sports_entropy']

        user_behavioral = user_behavioral.merge(sports_entropy, on='user_id', how='left')

        # Segmentation RFM
        user_behavioral = self._create_rfm_segmentation(user_behavioral)

        self.user_behavioral_features = user_behavioral
        logger.info(f"Features comportementales créées pour {len(user_behavioral)} utilisateurs")

        return user_behavioral

    def _create_rfm_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée une segmentation RFM sophistiquée avec gestion robuste des cas limites."""

        def safe_qcut(series, q=5, labels=None, reverse=False):
            """
            Version robuste de qcut qui gère les cas avec peu de valeurs uniques.
            """
            series = series.fillna(0)

            unique_vals = series.nunique(dropna=True)
            if unique_vals < 2:
                return pd.Series([1] * len(series), index=series.index)

            try:
                result = pd.qcut(series, q=q, labels=labels or range(1, q + 1), duplicates='drop')
            except ValueError as e:
                try:
                    result = pd.cut(series, bins=q, labels=labels or range(1, q + 1), duplicates='drop')
                except Exception:
                    return pd.Series([1] * len(series), index=series.index)

            if reverse and hasattr(result, 'cat'):
                result = result.cat.rename_categories(lambda x: q + 1 - int(x))

            return result

        # Fréquence
        if 'bet_count_30d' in df.columns:
            df['frequency_score'] = safe_qcut(df['bet_count_30d'].fillna(0), q=5)

        # Montant
        if 'total_stake_30d' in df.columns:
            df['monetary_score'] = safe_qcut(df['total_stake_30d'].fillna(0), q=5)

        # Récence basée sur le dernier pari
        last_bet_days = self.hierarchical_data.groupby('user_id')['bet_timestamp'].max()
        last_bet_days = (pd.Timestamp.now() - last_bet_days).dt.days
        df = df.merge(last_bet_days.reset_index().rename(columns={'bet_timestamp': 'recency_days'}),
                      on='user_id', how='left')

        df['recency_score'] = safe_qcut(df['recency_days'].fillna(999), q=5, reverse=True)

        # Score RFM composite (en string)
        df['rfm_score'] = (
                df['recency_score'].astype(str) +
                df['frequency_score'].astype(str) +
                df['monetary_score'].astype(str)
        )

        # Segmentation utilisateur
        def rfm_segment(rfm):
            if pd.isna(rfm):
                return 'Unknown'
            champions = {'555', '554', '544', '545', '454', '455', '445'}
            loyal = {'543', '444', '435', '355', '354', '345', '344', '335'}
            potential = {'512', '511', '422', '421', '412', '411', '311'}
            new = {'155', '154', '144', '214', '215', '115', '114'}
            promising = {'512', '511', '311', '411'}
            attention = {'133', '123', '132', '143', '244'}
            sleeping = {'214', '215', '115', '114'}
            at_risk = {'111', '112', '121', '131', '141', '151'}
            lost = set()

            if rfm in champions:
                return 'Champions'
            elif rfm in loyal:
                return 'Loyal Customers'
            elif rfm in potential:
                return 'Potential Loyalists'
            elif rfm in new:
                return 'New Customers'
            elif rfm in promising:
                return 'Promising'
            elif rfm in attention:
                return 'Need Attention'
            elif rfm in sleeping:
                return 'About to Sleep'
            elif rfm in at_risk:
                return 'At Risk'
            else:
                return 'Lost'

        df['user_segment'] = df['rfm_score'].apply(rfm_segment)

        return df

    def create_event_content_features(self) -> pd.DataFrame:
        """Crée des features de contenu pour les événements."""
        logger.info("Création des features de contenu pour les événements...")

        df_events = self._handle_missing_values(self.df_events.copy())

        # Encodage des variables catégorielles
        categorical_features = []
        categorical_cols = ['sport', 'competition', 'country', 'venue', 'event_status']

        for col in categorical_cols:
            if col in df_events.columns:
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(df_events[col].astype(str))
                categorical_features.append(encoded_values.reshape(-1, 1))
                self.encoders[f'event_{col}'] = encoder

        # Features numériques
        numeric_features = []
        numeric_cols = ['popularity_index', 'expected_goals_home', 'expected_goals_away',
                       'team_strength_home', 'team_strength_away']

        for col in numeric_cols:
            if col in df_events.columns:
                values = df_events[col].fillna(df_events[col].median())
                numeric_features.append(values.values.reshape(-1, 1))

        # Features dérivées des équipes
        if 'team_strength_home' in df_events.columns and 'team_strength_away' in df_events.columns:
            strength_diff = (df_events['team_strength_home'] - df_events['team_strength_away']).values.reshape(-1, 1)
            strength_total = (df_events['team_strength_home'] + df_events['team_strength_away']).values.reshape(-1, 1)
            numeric_features.extend([strength_diff, strength_total])

        # Features textuelles
        text_features = None
        if self.config.text_analysis:
            # Création du texte combiné
            text_cols = ['teams', 'competition', 'venue']
            available_text_cols = [col for col in text_cols if col in df_events.columns]

            if available_text_cols:
                text_data = df_events[available_text_cols].fillna('').apply(
                    lambda row: ' '.join(row.astype(str)), axis=1
                )

                tfidf = TfidfVectorizer(
                    max_features=self.config.tfidf_max_features,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2
                )
                text_features = tfidf.fit_transform(text_data)
                self.encoders['event_tfidf'] = tfidf

        # Features temporelles spécifiques aux événements
        temporal_features = []
        if 'event_start_time' in df_events.columns:
            start_times = pd.to_datetime(df_events['event_start_time'])

            # Features temporelles
            hour_features = start_times.dt.hour.values.reshape(-1, 1)
            dow_features = start_times.dt.dayofweek.values.reshape(-1, 1)
            month_features = start_times.dt.month.values.reshape(-1, 1)

            temporal_features.extend([hour_features, dow_features, month_features])

        # Assemblage des features
        all_features = []

        if numeric_features:
            numeric_matrix = np.hstack(numeric_features + temporal_features)
            numeric_matrix = self.scalers['robust'].fit_transform(numeric_matrix)
            all_features.append(csr_matrix(numeric_matrix))

        if categorical_features:
            categorical_matrix = np.hstack(categorical_features)
            all_features.append(csr_matrix(categorical_matrix))

        if text_features is not None:
            all_features.append(text_features)

        # Combinaison finale
        if all_features:
            self.event_features = hstack(all_features)
        else:
            raise ValueError("Aucune feature d'événement n'a pu être créée")

        self.event_ids = df_events['event_id'].values
        logger.info(f"Features d'événements créées: {self.event_features.shape}")

        return self.event_features

    def create_market_outcome_features(self) -> pd.DataFrame:
        """Crée des features spécifiques aux marchés et outcomes."""
        logger.info("Création des features marchés/outcomes...")

        # Merge markets + outcomes
        market_outcome = self.df_markets.merge(self.df_outcomes, on='market_id', how='inner')

        # Features des marchés
        market_types = market_outcome['market_name'].value_counts()
        market_outcome['market_popularity'] = market_outcome['market_name'].map(market_types)

        # Encodage des types de marchés
        market_encoder = LabelEncoder()
        market_outcome['market_type_encoded'] = market_encoder.fit_transform(market_outcome['market_name'])
        self.encoders['market_type'] = market_encoder

        # Features des outcomes
        outcome_encoder = LabelEncoder()
        market_outcome['outcome_type_encoded'] = outcome_encoder.fit_transform(market_outcome['outcome_name'])
        self.encoders['outcome_type'] = outcome_encoder

        # Statistiques par marché depuis les paris
        if hasattr(self, 'df_bets') and not self.df_bets.empty:
            # Jointure avec bets pour calculer les stats
            bets_with_outcomes = self.df_bets.merge(market_outcome, on='outcome_id', how='left')

            market_stats = bets_with_outcomes.groupby('market_id').agg({
                'bet_amount': ['count', 'sum', 'mean'],
                'odds_used': ['mean', 'std'],
                'outcome': ['mean', 'sum']
            }).reset_index()

            market_stats.columns = ['market_id', 'market_bet_count', 'market_volume',
                                  'market_avg_bet', 'market_avg_odds', 'market_odds_std',
                                  'market_win_rate', 'market_total_wins']

            market_outcome = market_outcome.merge(market_stats, on='market_id', how='left')

        # Normalisation des features numériques
        numeric_cols = [col for col in market_outcome.columns if market_outcome[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            market_outcome[numeric_cols] = self.scalers['standard'].fit_transform(
                market_outcome[numeric_cols].fillna(0)
            )

        self.market_outcome_features = market_outcome
        logger.info(f"Features marchés/outcomes créées: {len(market_outcome)} entrées")

        return market_outcome

    def create_advanced_user_item_matrix(self) -> csr_matrix:
        """Crée une matrice utilisateur-item avancée avec pondération sophistiquée."""
        logger.info("Création de la matrice utilisateur-item avancée...")

        if self.hierarchical_data is None:
            self.create_hierarchical_data()

        # Préparation des interactions avec pondération
        interactions = self.hierarchical_data[
            self.hierarchical_data['bet_amount'].notna() &
            (self.hierarchical_data['bet_amount'] > 0)
        ].copy()

        # Calcul des scores d'interaction sophistiqués
        user_event_stats = interactions.groupby(['user_id', 'event_id']).agg({
            'bet_amount': ['count', 'sum', 'mean'],
            'odds_used': ['mean', 'std'],
            'outcome': ['sum', 'count'],
            'is_live_bet': 'mean',
            'bet_timing_hours_before': 'mean'
        }).reset_index()

        # Flatten columns
        user_event_stats.columns = ['user_id', 'event_id', 'bet_frequency', 'total_stake',
                                   'avg_stake', 'avg_odds', 'odds_std', 'wins', 'total_bets',
                                   'live_bet_ratio', 'avg_timing']

        # Calcul des scores normalisés
        user_event_stats['frequency_score'] = MinMaxScaler().fit_transform(
            user_event_stats[['bet_frequency']]
        ).flatten()

        user_event_stats['monetary_score'] = MinMaxScaler().fit_transform(
            user_event_stats[['total_stake']]
        ).flatten()

        user_event_stats['success_score'] = user_event_stats['wins'] / user_event_stats['total_bets']
        user_event_stats['engagement_score'] = user_event_stats['live_bet_ratio']

        # Score de timing (plus proche de l'événement = plus d'engagement)
        user_event_stats['timing_score'] = 1 / (1 + user_event_stats['avg_timing'].abs())

        # Score composite final
        weights = {
            'frequency': 0.25,
            'monetary': 0.25,
            'success': 0.30,
            'engagement': 0.10,
            'timing': 0.10
        }

        user_event_stats['composite_rating'] = (
            weights['frequency'] * user_event_stats['frequency_score'] +
            weights['monetary'] * user_event_stats['monetary_score'] +
            weights['success'] * user_event_stats['success_score'] +
            weights['engagement'] * user_event_stats['engagement_score'] +
            weights['timing'] * user_event_stats['timing_score']
        ) * 5  # Échelle 0-5

        # Création des mappings
        unique_users = sorted(user_event_stats['user_id'].unique())
        unique_events = sorted(user_event_stats['event_id'].unique())

        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.event_id_mapping = {event_id: idx for idx, event_id in enumerate(unique_events)}
        self.user_idx_to_id = {idx: user_id for user_id, idx in self.user_id_mapping.items()}
        self.event_idx_to_id = {idx: event_id for event_id, idx in self.event_id_mapping.items()}

        # Construction de la matrice creuse
        user_indices = [self.user_id_mapping[uid] for uid in user_event_stats['user_id']]
        event_indices = [self.event_id_mapping[eid] for eid in user_event_stats['event_id']]
        ratings = user_event_stats['composite_rating'].values

        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, event_indices)),
            shape=(len(unique_users), len(unique_events))
        )

        # Statistiques de la matrice
        sparsity = 1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape)
        logger.info(f"Matrice utilisateur-item créée: {self.user_item_matrix.shape}, "
                   f"sparsité: {sparsity:.4f}, interactions: {self.user_item_matrix.nnz}")

        return self.user_item_matrix

    def create_feast_features(self):
        """Crée et enregistre les features dans Feast."""
        if not self.config.use_feast or self.feast_store is None:
            logger.warning("Feast non configuré")
            return

        logger.info("Création des feature views Feast...")

        try:
            # Entités
            user_entity = Entity(name="user_id", value_type=ValueType.INT64)
            event_entity = Entity(name="event_id", value_type=ValueType.INT64)
            market_entity = Entity(name="market_id", value_type=ValueType.INT64)

            # User Profile Features
            if hasattr(self, 'user_behavioral_features'):
                user_source = FileSource(
                    path="data/user_features.parquet",
                    timestamp_field="feature_timestamp"
                )

                user_profile_view = FeatureView(
                    name="user_profile_features",
                    entities=["user_id"],
                    ttl=timedelta(days=90),
                    features=[
                        Feature(name="avg_stake_7d", dtype=ValueType.FLOAT),
                        Feature(name="win_rate_7d", dtype=ValueType.FLOAT),
                        Feature(name="sports_entropy", dtype=ValueType.FLOAT),
                        Feature(name="user_segment", dtype=ValueType.STRING),
                        Feature(name="rfm_score", dtype=ValueType.STRING)
                    ],
                    online=self.config.online_store_enabled,
                    source=user_source,
                )

                # Sauvegarder les features utilisateur
                self.user_behavioral_features['feature_timestamp'] = pd.Timestamp.now()
                self.user_behavioral_features.to_parquet("data/user_features.parquet")

            # Event Features
            if hasattr(self, 'event_features'):
                event_source = FileSource(
                    path="data/event_features.parquet",
                    timestamp_field="feature_timestamp"
                )

                event_view = FeatureView(
                    name="event_features",
                    entities=["event_id"],
                    ttl=timedelta(days=30),
                    features=[
                        Feature(name="popularity_index", dtype=ValueType.FLOAT),
                        Feature(name="team_strength_diff", dtype=ValueType.FLOAT),
                        Feature(name="event_live", dtype=ValueType.BOOL),
                        Feature(name="days_until_event", dtype=ValueType.INT64)
                    ],
                    online=self.config.online_store_enabled,
                    source=event_source,
                )

            logger.info("Feature views Feast créés avec succès")

        except Exception as e:
            logger.error(f"Erreur création Feast features: {e}")

    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Gestion intelligente des valeurs manquantes.

        - Pour les colonnes catégorielles :
            * Si plus de 50 % de valeurs manquantes → remplissage par "Unknown".
            * Sinon → remplissage par la modalité la plus fréquente (mode).
        - Pour les colonnes numériques :
            * Si 'smart' : >50 % → 0 ; sinon → médiane.
            * Si autre stratégie : remplissage systématique par 0.
        """
        df = df.copy()

        for col in df.columns:
            missing_count = int(df[col].isnull().values.sum())  #df[col].isnull().sum().item()  # or int(df[col].isnull().sum())

            if missing_count > 0:
                missing_pct = missing_count / len(df) * 100

                if df[col].dtype.name in ['object', 'category']:
                    if missing_pct > 50:
                        df[col] = df[col].fillna('Unknown')
                    else:
                        mode_val = df[col].mode(dropna=True)
                        mode_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                else:
                    if strategy == 'smart':
                        if missing_pct > 50:
                            df[col] = df[col].fillna(0)
                        else:
                            median_val = df[col].median()
                            if pd.isna(median_val):
                                median_val = 0
                            df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0)

        return df

    def _detect_and_handle_outliers(self, data_frame: pd.DataFrame, numeric_cols: List[str],
                                    method: str = 'iqr') -> pd.DataFrame:
        df = data_frame.copy()

        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]

        for col in existing_numeric_cols:
            if df[col].notna().values.sum() == 0:
                continue

            if method == 'iqr':
                # Calculate quantiles. Ensure scalar result by trying .item() or .iloc[0]
                # Default to NaN if calculation is problematic (e.g., column is empty after filtering)
                try:
                    Q1_val = df[col].quantile(0.25).item()
                except ValueError: # Catches "can only convert an array of size 1 to a Python scalar" if .quantile() returns a Series with >1 element
                    Q1_val = np.nan
                except AttributeError: # Catches if .quantile() returns a scalar directly and .item() is called on it
                    Q1_val = df[col].quantile(0.25)

                try:
                    Q3_val = df[col].quantile(0.75).item()
                except ValueError:
                    Q3_val = np.nan
                except AttributeError:
                    Q3_val = df[col].quantile(0.75)

                Q1 = Q1_val if pd.notna(Q1_val) else 0.0
                Q3 = Q3_val if pd.notna(Q3_val) else 0.0

                IQR = Q3 - Q1

                if IQR == 0:
                    continue

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()

                if std_val == 0:
                    continue

                z_scores = np.abs((df[col] - mean_val) / std_val)
                median_val = df[col].median()

                df.loc[z_scores > 3, col] = median_val

        return df

    def prepare_comprehensive_training_data(
            self,
            target_column: str = 'outcome',
            include_temporal: bool = True,
            include_behavioral: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prépare un dataset d'entraînement complet avec toutes les features."""
        logger.info("Préparation du dataset d'entraînement complet...")

        try:
            # 1. Initialisation des données hiérarchiques
            if self.hierarchical_data is None:
                self.create_hierarchical_data()

            if include_behavioral and not hasattr(self, 'user_behavioral_features'):
                self.create_user_behavioral_features()

            if not hasattr(self, 'market_outcome_features'):
                self.create_market_outcome_features()

            # 2. Filtrage des données valides pour entraînement
            training_data = self.hierarchical_data[
                self.hierarchical_data['bet_status'].isin(['Won', 'Lost']) &
                self.hierarchical_data[target_column].notna()
                ].copy()

            if training_data.empty:
                raise ValueError("Aucune donnée d'entraînement disponible")

            logger.info(f"Données d'entraînement filtrées: {len(training_data)} lignes")

            # 3. Merge avec les features comportementales
            if include_behavioral and hasattr(self, 'user_behavioral_features'):
                training_data = training_data.merge(
                    self.user_behavioral_features, on='user_id', how='left'
                )
                logger.info("Features comportementales ajoutées")

            # 4. Merge avec les features marché/outcomes
            if hasattr(self, 'market_outcome_features'):
                market_features = self.market_outcome_features[
                    ['outcome_id', 'market_type_encoded', 'outcome_type_encoded', 'market_popularity']
                ]
                training_data = training_data.merge(
                    market_features,
                    on='outcome_id',
                    how='left'
                )
                logger.info("Features marché/outcomes ajoutées")

            # 5. Sélection des groupes de features
            feature_groups = {
                'user_profile': ['age', 'total_deposits', 'vip_status', 'days_since_registration'],
                'user_behavior': [col for col in training_data.columns if
                                  any(suffix in col for suffix in ['_7d', '_30d', '_90d', 'entropy', 'segment'])],
                'event_context': ['sport', 'competition', 'popularity_index', 'days_until_event', 'hours_until_event'],
                'temporal': [col for col in training_data.columns if
                             any(suffix in col for suffix in ['_sin', '_cos', 'hour', 'day_of_week', 'weekend'])],
                'bet_context': ['bet_amount', 'odds_used', 'is_live_bet', 'bet_timing_hours_before'],
                'market': ['market_type_encoded', 'outcome_type_encoded', 'market_popularity']
            }

            selected_features = []
            for group, features in feature_groups.items():
                if group == 'temporal' and not include_temporal:
                    continue
                if group == 'user_behavior' and not include_behavioral:
                    continue

                # Vérifier que les features existent dans le DataFrame
                existing_features = [f for f in features if f in training_data.columns]
                selected_features.extend(existing_features)

                if existing_features:
                    logger.info(f"Groupe {group}: {len(existing_features)} features ajoutées")

            if not selected_features:
                raise ValueError("Aucune feature sélectionnée")

            # 6. Encodage des colonnes catégorielles restantes
            categorical_cols = training_data[selected_features].select_dtypes(include=['object', 'category']).columns

            for col in categorical_cols:
                if col not in self.encoders:
                    encoder = LabelEncoder()
                    training_data[col] = training_data[col].fillna('Unknown')
                    training_data[col] = encoder.fit_transform(training_data[col].astype(str))
                    self.encoders[col] = encoder
                else:
                    training_data[col] = training_data[col].fillna('Unknown')
                    try:
                        training_data[col] = self.encoders[col].transform(training_data[col].astype(str))
                    except ValueError:
                        mask = ~training_data[col].isin(self.encoders[col].classes_)
                        training_data.loc[mask, col] = 'Unknown'
                        training_data[col] = self.encoders[col].transform(training_data[col].astype(str))

            # 7. Création de features dérivées
            derived_features = []

            if {'bet_amount', 'total_deposits'}.issubset(training_data.columns):
                training_data['bet_ratio'] = training_data['bet_amount'] / (training_data['total_deposits'] + 1)
                derived_features.append('bet_ratio')

            if 'odds_used' in training_data.columns:

                odds_safe = training_data['odds_used'].clip(lower=1.01)
                training_data['implied_probability'] = 1 / odds_safe
                training_data['odds_log'] = np.log(odds_safe)
                derived_features.extend(['implied_probability', 'odds_log'])

            if {'age', 'days_since_registration'}.issubset(training_data.columns):
                training_data['experience_ratio'] = training_data['days_since_registration'] / (
                            training_data['age'] * 365 + 1)
                derived_features.append('experience_ratio')

            # 8. Features finales
            final_features = [f for f in (selected_features + derived_features) if f in training_data.columns]

            if not final_features:
                raise ValueError("Aucune feature finale disponible")

            logger.info(f"Features finales sélectionnées: {len(final_features)}")

            # 9. Extraction des X/y
            X = training_data[final_features].copy()
            y = training_data[target_column].copy()

            # 10. Vérification des types de données avant traitement
            logger.info(f"Shape de X avant traitement: {X.shape}")
            logger.info(f"Types de colonnes: {X.dtypes.value_counts().to_dict()}")

            # 11. Traitement des valeurs manquantes
            X = self._handle_missing_values(X, strategy='smart')
            logger.info("Valeurs manquantes traitées")

            # 12. Traitement des outliers si configuré
            if hasattr(self.config, 'handle_outliers') and self.config.handle_outliers:
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    X = self._detect_and_handle_outliers(X, numeric_cols)
                    logger.info("Outliers traités")


            X = X.replace([np.inf, -np.inf], np.nan)
            X = self._handle_missing_values(X, strategy='smart')

            # 14. Normalisation
            if not hasattr(self, 'scalers'):
                self.scalers = {}

            if 'robust' not in self.scalers:
                from sklearn.preprocessing import RobustScaler
                self.scalers['robust'] = RobustScaler()

            # Vérification finale avant normalisation
            if X.shape[1] == 0:
                raise ValueError("Aucune feature disponible pour la normalisation")

            X_scaled = self.scalers['robust'].fit_transform(X)
            logger.info("Normalisation appliquée")

            # 15. Stockage des noms de features
            if not hasattr(self, 'feature_names'):
                self.feature_names = {}
            self.feature_names['final_training'] = final_features

            # 16. Validation finale
            if np.any(np.isnan(X_scaled)):
                logger.warning("Des valeurs NaN persistent dans X_scaled - nettoyage final")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            if np.any(np.isinf(X_scaled)):
                logger.warning("Des valeurs infinies persistent dans X_scaled - nettoyage final")
                X_scaled = np.nan_to_num(X_scaled, posinf=1.0, neginf=-1.0)

            logger.info(f" Dataset d'entraînement préparé: {X_scaled.shape[0]} exemples, {X_scaled.shape[1]} features")
            logger.info(f" Distribution du target: {y.value_counts().to_dict()}")

            return X_scaled, y.values, final_features

        except Exception as e:
            logger.error(f"Erreur dans prepare_comprehensive_training_data: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet de tous les traitements effectués."""
        summary = {
            'data_overview': {
                'users': len(self.df_users),
                'events': len(self.df_events),
                'markets': len(self.df_markets),
                'outcomes': len(self.df_outcomes),
                'bets': len(self.df_bets),
                'hierarchical_entries': len(self.hierarchical_data) if self.hierarchical_data is not None else 0
            },
            'feature_matrices': {
                'user_item_matrix_shape': self.user_item_matrix.shape if self.user_item_matrix is not None else None,
                'event_features_shape': self.event_features.shape if self.event_features is not None else None,
                'user_behavioral_features': len(self.user_behavioral_features) if hasattr(self, 'user_behavioral_features') else 0,
                'market_outcome_features': len(self.market_outcome_features) if hasattr(self, 'market_outcome_features') else 0
            },
            'processing_artifacts': {
                'encoders_count': len(self.encoders),
                'scalers_available': list(self.scalers.keys()),
                'feature_groups': self.feature_names
            },
            'configuration': self.config.__dict__,
            'feast_integration': {
                'enabled': self.config.use_feast,
                'store_configured': self.feast_store is not None
            }
        }

        return summary

    def save_all_artifacts(self, base_path: str = "./artifacts"):
        """Sauvegarde tous les artefacts de processing."""
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)

        # Modèles et encodeurs
        artifacts = {
            'encoders': self.encoders,
            'scalers': self.scalers,
            'mappings': {
                'user_id_mapping': self.user_id_mapping,
                'event_id_mapping': self.event_id_mapping,
                'market_id_mapping': getattr(self, 'market_id_mapping', {}),
                'outcome_id_mapping': getattr(self, 'outcome_id_mapping', {})
            },
            'feature_names': self.feature_names,
            'config': self.config
        }

        with open(base_path / 'processing_artifacts.pkl', 'wb') as f:
            pickle.dump(artifacts, f)

        # Matrices et features
        if self.user_item_matrix is not None:
            joblib.dump(self.user_item_matrix, base_path / 'user_item_matrix.pkl')

        if self.event_features is not None:
            joblib.dump(self.event_features, base_path / 'event_features.pkl')

        # DataFrames traités
        if hasattr(self, 'user_behavioral_features'):
            self.user_behavioral_features.to_parquet(base_path / 'user_behavioral_features.parquet')

        if hasattr(self, 'market_outcome_features'):
            self.market_outcome_features.to_parquet(base_path / 'market_outcome_features.parquet')

        if self.hierarchical_data is not None:
            # Échantillon pour éviter les gros fichiers
            sample_size = min(100000, len(self.hierarchical_data))
            self.hierarchical_data.sample(n=sample_size).to_parquet(base_path / 'hierarchical_data_sample.parquet')

        logger.info(f"Tous les artefacts sauvegardés dans {base_path}")

    def load_artifacts(self, base_path: str = "./artifacts"):
        """Charge les artefacts de processing."""
        base_path = Path(base_path)

        # Artefacts principaux
        with open(base_path / 'processing_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)

        self.encoders = artifacts['encoders']
        self.scalers = artifacts['scalers']
        self.user_id_mapping = artifacts['mappings']['user_id_mapping']
        self.event_id_mapping = artifacts['mappings']['event_id_mapping']
        self.market_id_mapping = artifacts['mappings'].get('market_id_mapping', {})
        self.outcome_id_mapping = artifacts['mappings'].get('outcome_id_mapping', {})
        self.feature_names = artifacts['feature_names']
        self.config = artifacts['config']

        # Matrices
        if (base_path / 'user_item_matrix.pkl').exists():
            self.user_item_matrix = joblib.load(base_path / 'user_item_matrix.pkl')

        if (base_path / 'event_features.pkl').exists():
            self.event_features = joblib.load(base_path / 'event_features.pkl')

        logger.info(f"Artefacts chargés depuis {base_path}")


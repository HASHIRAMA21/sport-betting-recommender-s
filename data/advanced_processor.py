import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from dataclasses import dataclass, field
from scipy.stats import entropy
import json
import pickle
from pathlib import Path
import joblib
from feast import Entity, Feature, FeatureView, ValueType, FileSource

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class ProcessingConfig:
    """Configuration avancée pour le préprocessing des données de paris sportifs."""

    # Feature Engineering
    tfidf_max_features: int = 300
    pca_components: int = 100
    kmeans_clusters: int = 15
    rolling_windows: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 30, 90])

    # Analysis flags
    text_analysis: bool = True
    feature_selection: bool = True
    handle_outliers: bool = True
    create_odds_features: bool = True
    create_temporal_features: bool = True
    create_market_features: bool = True

    # Thresholds
    outlier_threshold: float = 1.5
    min_market_occurrences: int = 10
    correlation_threshold: float = 0.95
    feature_importance_threshold: float = 0.001

    # Feast/Feature Store config
    use_feast: bool = False
    feast_repo_path: str = "./feature_repo"
    online_store_enabled: bool = True

    # Performance
    n_jobs: int = -1
    random_state: int = 42

class AdvancedDataProcessor:
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
        Crée une structure hiérarchique enrichie :
        - Classique : Event → Market → Outcome → Bet → User
        - Alternative : Event → Bet (directe via event_id) → User
        - Stocke les deux vues + une version combinée (dédoublonnée)
        """
        logger.info("Création de la structure hiérarchique des données...")

        try:

            def clean_text_column(df: pd.DataFrame, col: str):
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.lower()

            clean_text_column(self.df_markets, 'market_name')
            clean_text_column(self.df_outcomes, 'outcome_name')
            clean_text_column(self.df_bets, 'market')
            clean_text_column(self.df_bets, 'outcome')

            events_markets = self.df_events.merge(
                self.df_markets,
                on='event_id',
                how='left',
                suffixes=('_event', '_market')
            )
            logger.info(f"Events + Markets joints: {len(events_markets)} lignes")

            if 'market_id' in events_markets.columns and 'market_id' in self.df_outcomes.columns:
                markets_outcomes = events_markets.merge(
                    self.df_outcomes,
                    on='market_id',
                    how='left',
                    suffixes=('', '_outcome')
                )
                logger.info(f"Markets + Outcomes joints: {len(markets_outcomes)} lignes")
            else:
                logger.warning("Colonne 'market_id' manquante. Outcomes non joints.")
                markets_outcomes = events_markets

            required_bet_cols = {'event_id', 'market', 'outcome'}
            if required_bet_cols.issubset(self.df_bets.columns):
                full_classic = markets_outcomes.merge(
                    self.df_bets,
                    left_on=['event_id', 'market_name', 'outcome_name'],
                    right_on=['event_id', 'market', 'outcome'],
                    how='left',
                    suffixes=('', '_bet')
                )
                logger.info(f"Jointure classique complète (via market/outcome): {len(full_classic)} lignes")
            else:
                missing_cols = required_bet_cols - set(self.df_bets.columns)
                logger.warning(f"Colonnes manquantes pour jointure classique : {missing_cols}")
                full_classic = markets_outcomes

            if 'event_id' in self.df_events.columns and 'event_id' in self.df_bets.columns:
                # Préserver la colonne 'id' de bet
                df_bets_renamed = self.df_bets.rename(columns={'id': 'bet_id'})
                direct_join = self.df_events.merge(
                    df_bets_renamed,
                    on='event_id',
                    how='left',
                    suffixes=('_event', '_bet')
                )
                logger.info(f"Jointure directe Events + Bets : {len(direct_join)} lignes")
            else:
                logger.warning("Jointure directe Events → Bets impossible (event_id manquant)")
                direct_join = pd.DataFrame()

            # --- 5. Ajout des utilisateurs dans les deux cas ---
            def enrich_with_users(df: pd.DataFrame) -> pd.DataFrame:
                if 'user_id' in df.columns and 'user_id' in self.df_users.columns:
                    df = df.merge(self.df_users, on='user_id', how='left')
                    logger.info(f"Utilisateurs associés : {len(df)} lignes")
                else:
                    logger.warning("Colonne 'user_id' manquante pour jointure utilisateurs.")
                return df

            full_classic = enrich_with_users(full_classic)
            direct_join = enrich_with_users(direct_join)

            combined = pd.concat([full_classic, direct_join], ignore_index=True)

            dedup_key = 'id' if 'id' in combined.columns else (
                'bet_id' if 'bet_id' in combined.columns else None
            )

            if dedup_key:
                combined = combined.drop_duplicates(subset=[dedup_key], keep='first')
                logger.info(f"Dédoublonnage effectué sur '{dedup_key}'")
            else:
                logger.warning("Aucune colonne d'identifiant trouvée pour le dédoublonnage.")

            if hasattr(self, '_enrich_temporal_data'):
                self._enrich_temporal_data(combined)

            if hasattr(self, 'df_odds_history') and self.df_odds_history is not None:
                self._enrich_odds_data(combined)

            if hasattr(self, '_create_hierarchy_derived_features'):
                needed_cols = {'bet_amount', 'odds_used', 'event_id'}
                if needed_cols.issubset(combined.columns):
                    self._create_hierarchy_derived_features(combined)
                else:
                    logger.warning(
                        f"Colonnes manquantes pour features dérivées : {needed_cols - set(combined.columns)}")

            # --- 9. Stockage dans l'objet ---
            self.hierarchical_data = combined
            self.hierarchical_data_classic = full_classic
            self.hierarchical_data_direct = direct_join

            logger.info(f"Structure hiérarchique combinée créée avec succès : {len(combined)} lignes")

            return combined

        except Exception as e:
            logger.error(f"Erreur dans la création de la structure hiérarchique : {e}")
            raise



    def _enrich_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit le DataFrame avec des features temporelles avancées pour les événements et les paris.

        Args:
            df (pd.DataFrame): DataFrame contenant au moins les colonnes temporelles si présentes.

        Returns:
            pd.DataFrame: DataFrame enrichi avec des features temporelles.
        """
        logger.info("Enrichissement des données temporelles...")

        # Colonnes timestamps à convertir
        timestamp_columns = ['event_start_time', 'bet_timestamp', 'market_open_time']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        now = pd.Timestamp.now()

        # Features liées aux événements
        if 'event_start_time' in df.columns:
            df['days_until_event'] = (df['event_start_time'] - now).dt.days
            df['hours_until_event'] = (df['event_start_time'] - now).dt.total_seconds() / 3600
            df['event_month'] = df['event_start_time'].dt.month
            df['event_day_of_week'] = df['event_start_time'].dt.dayofweek
            df['event_hour'] = df['event_start_time'].dt.hour
            df['is_weekend_event'] = (df['event_day_of_week'] >= 5).astype(int)

            # Features cycliques pour gérer la périodicité temporelle
            df['event_hour_sin'] = np.sin(2 * np.pi * df['event_hour'] / 24)
            df['event_hour_cos'] = np.cos(2 * np.pi * df['event_hour'] / 24)
            df['event_month_sin'] = np.sin(2 * np.pi * df['event_month'] / 12)
            df['event_month_cos'] = np.cos(2 * np.pi * df['event_month'] / 12)

        # Features liées aux paris
        if 'bet_timestamp' in df.columns:
            df['bet_hour'] = df['bet_timestamp'].dt.hour
            df['bet_day_of_week'] = df['bet_timestamp'].dt.dayofweek
            df['is_bet_weekend'] = (df['bet_day_of_week'] >= 5).astype(int)

            # Calcul du timing du pari par rapport à l'événement (en heures)
            if 'event_start_time' in df.columns:
                df['bet_timing_hours_before'] = (df['event_start_time'] - df['bet_timestamp']).dt.total_seconds() / 3600
                df['is_live_bet'] = (df['bet_timing_hours_before'] <= 0).astype(int)
            else:
                df['bet_timing_hours_before'] = pd.NA
                df['is_live_bet'] = pd.NA

        logger.info("Enrichissement temporel terminé.")

        return df

    def _enrich_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit le DataFrame principal avec des statistiques historiques sur les cotes (odds).

        Args:
            df (pd.DataFrame): DataFrame principal contenant au moins la colonne 'outcome_id'.

        Returns:
            pd.DataFrame: DataFrame enrichi avec les features d'historique des cotes.
        """
        logger.info("Enrichissement avec l'historique des cotes...")

        if self.df_odds_history is None or self.df_odds_history.empty:
            logger.warning("Historique des cotes manquant ou vide, enrichissement ignoré.")
            return df

        required_cols = {'outcome_id', 'odds_value', 'timestamp'}
        missing_cols = required_cols - set(self.df_odds_history.columns)
        if missing_cols:
            logger.error(f"Colonnes manquantes dans df_odds_history : {missing_cols}")
            return df

        odds_stats = self.df_odds_history.groupby('outcome_id').agg(
            odds_min=('odds_value', 'min'),
            odds_max=('odds_value', 'max'),
            odds_mean=('odds_value', 'mean'),
            odds_std=('odds_value', 'std'),
            odds_updates_count=('odds_value', 'count'),
            first_odds_time=('timestamp', 'min'),
            last_odds_time=('timestamp', 'max')
        ).reset_index()


        odds_stats['odds_volatility'] = odds_stats.apply(
            lambda row: (row['odds_std'] / row['odds_mean']) if row['odds_mean'] and row['odds_std'] else 0, axis=1)
        odds_stats['odds_range'] = odds_stats['odds_max'] - odds_stats['odds_min']
        odds_stats['odds_drift'] = odds_stats.apply(
            lambda row: ((row['odds_max'] - row['odds_min']) / row['odds_min']) if row['odds_min'] else 0, axis=1)

        df = df.merge(odds_stats, on='outcome_id', how='left')

        logger.info(f"Enrichissement effectué : {len(odds_stats)} outcomes traités.")

        return df

    def _create_hierarchy_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features dérivées à partir de la structure hiérarchique,
        au niveau des événements et des marchés, et les merge dans le DataFrame.

        Args:
            df (pd.DataFrame): DataFrame hiérarchique complet avec colonnes nécessaires.

        Returns:
            pd.DataFrame: DataFrame enrichi avec les features dérivées.
        """

        import numpy as np

        logger.info("Création des features dérivées hiérarchiques...")

        # Colonnes requises pour le calcul
        required_cols = {'event_id', 'market_id', 'outcome_id', 'bet_amount', 'odds_used'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"Colonnes manquantes dans le DataFrame pour features dérivées: {missing_cols}")
            return df

        # --- Features au niveau Event ---
        event_agg = df.groupby('event_id').agg(
            num_markets=('market_id', 'nunique'),
            num_outcomes=('outcome_id', 'nunique'),
            total_bets=('bet_amount', 'count'),
            total_volume=('bet_amount', 'sum'),
            avg_bet_amount=('bet_amount', 'mean'),
            avg_odds=('odds_used', 'mean'),
            odds_std=('odds_used', 'std')
        ).reset_index()

        # Ajouter features dérivées simples
        event_agg['event_popularity'] = event_agg['total_bets']
        event_agg['event_liquidity'] = event_agg['total_volume']

        # --- Features au niveau Market ---
        market_agg = df.groupby('market_id').agg(
            market_num_outcomes=('outcome_id', 'nunique'),
            market_bet_count=('bet_amount', 'count'),
            market_volume=('bet_amount', 'sum'),
            market_avg_bet=('bet_amount', 'mean'),
            market_avg_odds=('odds_used', 'mean'),
            market_odds_std=('odds_used', 'std')
        ).reset_index()

        # Merge des features Event et Market dans le DataFrame original
        df = df.merge(event_agg, on='event_id', how='left')
        df = df.merge(market_agg, on='market_id', how='left')

        logger.info(f"Features dérivées hiérarchiques ajoutées : {len(df)} lignes")

        return df

    def create_user_behavioral_features(self, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Crée des features comportementales avancées pour les utilisateurs,
        sur plusieurs fenêtres temporelles définies dans self.config.rolling_windows.
        Si aucune interaction dans une fenêtre, fait un fallback sur la totalité des données.

        Args:
            reference_date (pd.Timestamp, optional): Date de référence pour les fenêtres glissantes.
                                                    Par défaut, pd.Timestamp.now().

        Returns:
            pd.DataFrame: DataFrame des features comportementales par utilisateur.
        """
        if self.hierarchical_data is None:
            self.create_hierarchical_data()

        df = self.hierarchical_data.copy()
        if df.empty:
            logger.warning("Hierarchical data is empty, no behavioral features created.")
            return pd.DataFrame()

        if reference_date is None:
            reference_date = pd.Timestamp.now()

        required_cols = {'user_id', 'bet_timestamp', 'bet_amount', 'odds_used', 'outcome',
                         'sport', 'market_name', 'is_live_bet'}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Missing columns for behavioral features: {missing}")
            return pd.DataFrame()

        user_features = {}

        min_date = df['bet_timestamp'].min()

        for window in self.config.rolling_windows:
            start_date = reference_date - pd.Timedelta(days=window)
            window_data = df[df['bet_timestamp'] >= start_date]

            if window_data.empty:
                logger.info(f"Aucune donnée dans la fenêtre {window} jours, fallback sur toutes les données.")
                window_data = df[df['bet_timestamp'] >= min_date]

                if window_data.empty:
                    logger.warning(
                        "Même le fallback ne retourne aucune donnée, création d’un DataFrame vide structuré.")
                    user_features[window] = pd.DataFrame(columns=['user_id'] + [
                        f"{feat}_{window}d" for feat in [
                            'bet_count', 'total_stake', 'avg_stake', 'stake_std',
                            'avg_odds', 'odds_std', 'min_odds', 'max_odds',
                            'total_wins', 'win_rate', 'sports_diversity', 'market_diversity', 'live_bet_rate'
                        ]
                    ])
                    continue

            agg_dict = {
                'bet_amount': ['count', 'sum', 'mean', 'std'],
                'odds_used': ['mean', 'std', 'min', 'max'],
                'outcome': ['sum', lambda x: ((x == 1).sum() / len(x)) if len(x) > 0 else 0],
                'sport': pd.Series.nunique,
                'market_name': pd.Series.nunique,
                'is_live_bet': 'mean'
            }

            agg_df = window_data.groupby('user_id').agg(agg_dict)

            agg_df.columns = [
                'bet_count', 'total_stake', 'avg_stake', 'stake_std',
                'avg_odds', 'odds_std', 'min_odds', 'max_odds',
                'total_wins', 'win_rate', 'sports_diversity', 'market_diversity', 'live_bet_rate'
            ]

            agg_df = agg_df.reset_index()
            agg_df = agg_df.add_suffix(f'_{window}d')
            agg_df.rename(columns={f'user_id_{window}d': 'user_id'}, inplace=True)

            num_cols = agg_df.select_dtypes(include='number').columns
            agg_df[num_cols] = agg_df[num_cols].fillna(0)

            user_features[window] = agg_df

        user_behavioral = user_features[self.config.rolling_windows[0]]
        for window in self.config.rolling_windows[1:]:
            user_behavioral = user_behavioral.merge(user_features[window], on='user_id', how='outer')

        # Features dérivées
        if {'win_rate_7d', 'win_rate_30d'}.issubset(user_behavioral.columns):
            user_behavioral['win_rate_trend'] = user_behavioral['win_rate_7d'] - user_behavioral['win_rate_30d']

        if {'avg_stake_7d', 'avg_stake_30d'}.issubset(user_behavioral.columns):
            user_behavioral['stake_trend'] = user_behavioral['avg_stake_7d'] / (user_behavioral['avg_stake_30d'] + 1)

        # Entropy diversité sportive (sur toutes les données)
        sports_entropy = (
            df.groupby('user_id')['sport']
            .apply(lambda x: entropy(x.value_counts(normalize=True)) if len(x) > 1 else 0)
            .reset_index(name='sports_entropy')
        )
        user_behavioral = user_behavioral.merge(sports_entropy, on='user_id', how='left')
        user_behavioral['sports_entropy'].fillna(0, inplace=True)

        # Segmentation RFM (existe déjà)
        user_behavioral = self._create_rfm_segmentation(user_behavioral)

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
        """Crée une matrice utilisateur-item avancée avec diagnostic complet."""
        logger.info("Création de la matrice utilisateur-item avancée...")

        if self.hierarchical_data is None:
            self.create_hierarchical_data()

        # Vérification des colonnes requises
        required_cols = ['bet_amount', 'user_id', 'event_id', 'odds_used', 'outcome', 'is_live_bet',
                         'bet_timing_hours_before']
        missing_cols = [col for col in required_cols if col not in self.hierarchical_data.columns]

        if missing_cols:
            logger.error(f"Colonnes manquantes: {missing_cols}")
            return self._create_empty_matrix()

        # Filtrage et nettoyage des données
        interactions = self._filter_and_clean_data()

        if interactions.empty:
            logger.warning("Aucune interaction valide trouvée")
            return self._create_empty_matrix()

        logger.info(f"Traitement de {len(interactions)} interactions")

        # Agrégation des statistiques par utilisateur/événement
        user_event_stats = self._aggregate_user_event_stats(interactions)

        if user_event_stats.empty:
            logger.warning("Aucune statistique utilisateur-événement générée")
            return self._create_empty_matrix()

        # Calcul des scores et création de la matrice
        user_event_stats = self._calculate_composite_scores(user_event_stats)
        return self._build_sparse_matrix(user_event_stats)

    def _filter_and_clean_data(self) -> pd.DataFrame:
        """Filtre et nettoie les données d'interaction."""
        # Filtres de base
        mask = (
                self.hierarchical_data['bet_amount'].notna() &
                self.hierarchical_data['user_id'].notna() &
                self.hierarchical_data['event_id'].notna()
        )

        interactions = self.hierarchical_data[mask].copy()

        if interactions.empty:
            return interactions

        # Conversion et nettoyage des types de données
        interactions = self._clean_data_types(interactions)

        # Filtre sur le montant des paris
        interactions = interactions[interactions['bet_amount'] > 0]

        return interactions

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et convertit les types de données."""
        df = df.copy()

        # Colonnes numériques à nettoyer
        numeric_columns = {
            'bet_amount': 0.0,
            'odds_used': 1.0,
            'outcome': 0,
            'bet_timing_hours_before': 24.0
        }

        for col, default_value in numeric_columns.items():
            if col in df.columns:
                # Conversion en numérique, remplacement des erreurs par NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(default_value)
            else:
                df[col] = default_value

        # Colonne booléenne
        if 'is_live_bet' in df.columns:
            df['is_live_bet'] = pd.to_numeric(df['is_live_bet'], errors='coerce').fillna(0).astype(bool)
        else:
            df['is_live_bet'] = False

        # S'assurer que outcome est entier
        df['outcome'] = df['outcome'].astype(int)

        return df

    def _aggregate_user_event_stats(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Agrège les statistiques par utilisateur et événement."""
        try:
            # Définition des agrégations avec gestion des erreurs
            agg_dict = {
                'bet_amount': ['count', 'sum', 'mean'],
                'odds_used': ['mean', 'std'],
                'outcome': ['sum', 'count'],
                'is_live_bet': 'mean',
                'bet_timing_hours_before': 'mean'
            }

            user_event_stats = interactions.groupby(['user_id', 'event_id']).agg(agg_dict).reset_index()

            # Aplatissement des colonnes multi-niveaux
            user_event_stats.columns = [
                'user_id', 'event_id', 'bet_frequency', 'total_stake', 'avg_stake',
                'avg_odds', 'odds_std', 'wins', 'total_bets', 'live_bet_ratio', 'avg_timing'
            ]

        except Exception as e:
            logger.warning(f"Erreur lors de l'agrégation, utilisation de statistiques simplifiées: {e}")
            # Fallback vers une agrégation simple
            user_event_stats = interactions.groupby(['user_id', 'event_id']).agg({
                'bet_amount': ['count', 'sum'],
                'outcome': 'sum'
            }).reset_index()

            user_event_stats.columns = ['user_id', 'event_id', 'bet_frequency', 'total_stake', 'wins']
            user_event_stats['avg_stake'] = user_event_stats['total_stake'] / user_event_stats['bet_frequency']
            user_event_stats['total_bets'] = user_event_stats['bet_frequency']
            user_event_stats['avg_odds'] = 1.0
            user_event_stats['odds_std'] = 0.0
            user_event_stats['live_bet_ratio'] = 0.0
            user_event_stats['avg_timing'] = 24.0

        user_event_stats = user_event_stats.fillna(0)

        numeric_cols = ['wins', 'total_bets', 'bet_frequency', 'total_stake', 'avg_stake',
                        'avg_odds', 'odds_std', 'live_bet_ratio', 'avg_timing']

        for col in numeric_cols:
            if col in user_event_stats.columns:
                user_event_stats[col] = pd.to_numeric(user_event_stats[col], errors='coerce').fillna(0)

        return user_event_stats

    def _calculate_composite_scores(self, user_event_stats: pd.DataFrame) -> pd.DataFrame:
        """Calcule les scores composites pour chaque interaction utilisateur-événement."""
        # Normalisation des scores de fréquence et monétaires
        if len(user_event_stats) > 1:
            scaler = MinMaxScaler()

            # Score de fréquence
            if user_event_stats['bet_frequency'].std() > 0:
                user_event_stats['frequency_score'] = scaler.fit_transform(
                    user_event_stats[['bet_frequency']]
                ).flatten()
            else:
                user_event_stats['frequency_score'] = 1.0

            # Score monétaire
            if user_event_stats['total_stake'].std() > 0:
                user_event_stats['monetary_score'] = scaler.fit_transform(
                    user_event_stats[['total_stake']]
                ).flatten()
            else:
                user_event_stats['monetary_score'] = 1.0
        else:
            user_event_stats['frequency_score'] = 1.0
            user_event_stats['monetary_score'] = 1.0

        # Score de succès (ratio de victoires)
        user_event_stats['success_score'] = np.where(
            user_event_stats['total_bets'] > 0,
            user_event_stats['wins'].astype(float) / user_event_stats['total_bets'].astype(float),
            0.0
        )

        # Score d'engagement (paris en direct)
        user_event_stats['engagement_score'] = user_event_stats['live_bet_ratio'].astype(float)

        # Score de timing (proximité avec l'événement)
        user_event_stats['timing_score'] = 1.0 / (1.0 + np.abs(user_event_stats['avg_timing'].astype(float)))

        # Gestion des valeurs NaN/inf
        score_columns = ['frequency_score', 'monetary_score', 'success_score', 'engagement_score', 'timing_score']
        for col in score_columns:
            user_event_stats[col] = user_event_stats[col].replace([np.inf, -np.inf], 0).fillna(0)

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
                                               ) * 5.0  # Échelle 0-5

        # Limitation des valeurs entre 0 et 5
        user_event_stats['composite_rating'] = np.clip(user_event_stats['composite_rating'], 0, 5)

        return user_event_stats

    def _build_sparse_matrix(self, user_event_stats: pd.DataFrame) -> csr_matrix:
        """Construit la matrice creuse utilisateur-item."""
        # Création des mappings
        unique_users = sorted(user_event_stats['user_id'].unique())
        unique_events = sorted(user_event_stats['event_id'].unique())

        self.user_id_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.event_id_mapping = {event_id: idx for idx, event_id in enumerate(unique_events)}
        self.user_idx_to_id = {idx: user_id for user_id, idx in self.user_id_mapping.items()}
        self.event_idx_to_id = {idx: event_id for event_id, idx in self.event_id_mapping.items()}

        try:
            # Construction de la matrice creuse
            user_indices = [self.user_id_mapping[uid] for uid in user_event_stats['user_id']]
            event_indices = [self.event_id_mapping[eid] for eid in user_event_stats['event_id']]
            ratings = user_event_stats['composite_rating'].values.astype(float)

            self.user_item_matrix = csr_matrix(
                (ratings, (user_indices, event_indices)),
                shape=(len(unique_users), len(unique_events))
            )

            # Statistiques finales
            sparsity = 1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape)
            logger.info(f"Matrice créée: {self.user_item_matrix.shape}, "
                        f"sparsité: {sparsity:.4f}, interactions: {self.user_item_matrix.nnz}")

            if ratings.size > 0:
                logger.info(f"Ratings: min={ratings.min():.3f}, max={ratings.max():.3f}, mean={ratings.mean():.3f}")

        except Exception as e:
            logger.error(f"Erreur lors de la création de la matrice: {e}")
            self.user_item_matrix = csr_matrix((len(unique_users), len(unique_events)))

        return self.user_item_matrix

    def _create_empty_matrix(self) -> csr_matrix:
        """Crée une matrice vide en cas d'échec."""
        self.user_item_matrix = csr_matrix((0, 0))
        self.user_id_mapping = {}
        self.event_id_mapping = {}
        self.user_idx_to_id = {}
        self.event_idx_to_id = {}
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
        """Traite les valeurs manquantes avec différentes stratégies."""
        # Vérification que df est bien un DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning(f"_handle_missing_values: df n'est pas un DataFrame, mais {type(df)}")
            # Tentative de conversion en DataFrame
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                logger.error(f"Impossible de convertir en DataFrame: {e}")
                return df

        # Vérification que df n'est pas vide
        if df.empty:
            return df

        df_copy = df.copy()

        try:
            # Séparation par type de données
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
            bool_cols = df_copy.select_dtypes(include=['bool']).columns.tolist()
            datetime_cols = df_copy.select_dtypes(include=['datetime64']).columns.tolist()

            if strategy == 'smart':
                # Colonnes numériques : médiane
                for col in numeric_cols:
                    # Vérification sécurisée pour éviter les comparaisons ambiguës
                    has_na = df_copy[col].isna().any()
                    if isinstance(has_na, bool) and has_na:
                        median_val = df_copy[col].median()
                        # Vérification que median_val est un scalaire
                        if hasattr(median_val, 'item'):
                            try:
                                median_val = median_val.item()
                            except:
                                pass
                        df_copy[col] = df_copy[col].fillna(median_val if pd.notna(median_val) else 0)

                # Colonnes catégorielles : mode ou 'Unknown'
                for col in categorical_cols:
                    has_na = df_copy[col].isna().any()
                    if isinstance(has_na, bool) and has_na:
                        mode_val = df_copy[col].mode()
                        if len(mode_val) > 0:
                            fill_val = mode_val.iloc[0]
                        else:
                            fill_val = 'Unknown'
                        df_copy[col] = df_copy[col].fillna(fill_val)

                # Colonnes booléennes : mode ou False
                for col in bool_cols:
                    has_na = df_copy[col].isna().any()
                    if isinstance(has_na, bool) and has_na:
                        mode_val = df_copy[col].mode()
                        if len(mode_val) > 0:
                            fill_val = mode_val.iloc[0]
                        else:
                            fill_val = False
                        df_copy[col] = df_copy[col].fillna(fill_val)

                # Colonnes datetime : forward fill
                for col in datetime_cols:
                    has_na = df_copy[col].isna().any()
                    if isinstance(has_na, bool) and has_na:
                        df_copy[col] = df_copy[col].fillna(method='ffill')

            elif strategy == 'zero':
                df_copy = df_copy.fillna(0)
            elif strategy == 'median':
                for col in numeric_cols:
                    median_val = df_copy[col].median()
                    # Vérification que median_val est un scalaire
                    if hasattr(median_val, 'item'):
                        try:
                            median_val = median_val.item()
                        except:
                            pass
                    df_copy[col] = df_copy[col].fillna(median_val if pd.notna(median_val) else 0)

                for col in categorical_cols + bool_cols:
                    mode_val = df_copy[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val.iloc[0]
                    else:
                        fill_val = 'Unknown' if col in categorical_cols else False
                    df_copy[col] = df_copy[col].fillna(fill_val)
            else:
                df_copy = df_copy.fillna(0)

            # Nettoyage final
            has_na = df_copy.isna().any().any()
            if isinstance(has_na, bool) and has_na:
                df_copy = df_copy.fillna(0)

            return df_copy

        except Exception as e:
            logger.error(f"Erreur dans _handle_missing_values: {str(e)}")
            return df.fillna(0)


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
            # 1. Initialisation des données
            if self.hierarchical_data is None:
                self.create_hierarchical_data()

            if self.hierarchical_data is None or self.hierarchical_data.empty:
                raise ValueError("Aucune donnée hiérarchique disponible")

            if include_behavioral and not hasattr(self, 'user_behavioral_features'):
                self.create_user_behavioral_features()

            if not hasattr(self, 'market_outcome_features'):
                self.create_market_outcome_features()

            # 2. Filtrage des données d'entraînement
            training_data = self.hierarchical_data[
                self.hierarchical_data['bet_status'].isin(['Won', 'Lost']) &
                self.hierarchical_data[target_column].notna()
                ].copy()

            if training_data.empty:
                # Tentative avec critères moins restrictifs
                training_data = self.hierarchical_data[
                    self.hierarchical_data[target_column].notna()
                ].copy()

                if training_data.empty:
                    raise ValueError("Aucune donnée d'entraînement disponible après filtrage")

            logger.info(f"Données d'entraînement: {len(training_data)} lignes")

            # 3. Merge des features comportementales
            if include_behavioral and hasattr(self,
                                              'user_behavioral_features') and not self.user_behavioral_features.empty:
                initial_rows = len(training_data)
                training_data = training_data.merge(
                    self.user_behavioral_features, on='user_id', how='left'
                )
                if len(training_data) != initial_rows:
                    logger.warning(
                        f"Perte de données lors du merge comportemental: {initial_rows} -> {len(training_data)}")

            # 4. Merge des features marché/outcomes
            if hasattr(self, 'market_outcome_features') and not self.market_outcome_features.empty:
                market_features = self.market_outcome_features[
                    [col for col in ['outcome_id', 'market_type_encoded', 'outcome_type_encoded', 'market_popularity']
                     if col in self.market_outcome_features.columns]
                ]
                if not market_features.empty and 'outcome_id' in training_data.columns:
                    training_data = training_data.merge(market_features, on='outcome_id', how='left')

            # 5. Définition des groupes de features
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

            # 6. Sélection des features existantes
            selected_features = []
            for group, features in feature_groups.items():
                if group == 'temporal' and not include_temporal:
                    continue
                if group == 'user_behavior' and not include_behavioral:
                    continue

                existing_features = [f for f in features if f in training_data.columns]
                selected_features.extend(existing_features)

            # Ajout de features de base si aucune n'est trouvée
            if not selected_features:
                basic_features = ['bet_amount', 'odds_used', 'user_id']
                selected_features = [f for f in basic_features if f in training_data.columns]

            if not selected_features:
                raise ValueError("Aucune feature utilisable trouvée")

            # 7. Encodage des variables catégorielles
            categorical_cols = training_data[selected_features].select_dtypes(include=['object', 'category']).columns

            if not hasattr(self, 'encoders'):
                self.encoders = {}

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
                        # Gestion des nouvelles catégories
                        mask = ~training_data[col].isin(self.encoders[col].classes_)
                        training_data.loc[mask, col] = 'Unknown'
                        training_data[col] = self.encoders[col].transform(training_data[col].astype(str))

            # 8. Création de features dérivées
            derived_features = []

            # Ratio de mise
            if {'bet_amount', 'total_deposits'}.issubset(training_data.columns):
                training_data['bet_ratio'] = training_data['bet_amount'] / (training_data['total_deposits'] + 1)
                derived_features.append('bet_ratio')

            # Features des cotes
            if 'odds_used' in training_data.columns:
                odds_safe = training_data['odds_used'].clip(lower=1.01, upper=1000)
                training_data['implied_probability'] = 1 / odds_safe
                training_data['odds_log'] = np.log(odds_safe)
                derived_features.extend(['implied_probability', 'odds_log'])

            # Ratio d'expérience
            if {'age', 'days_since_registration'}.issubset(training_data.columns):
                training_data['experience_ratio'] = training_data['days_since_registration'] / (
                        training_data['age'] * 365 + 1)
                derived_features.append('experience_ratio')

            # 9. Features finales
            final_features = [f for f in (selected_features + derived_features)
                              if f in training_data.columns and f != target_column]

            if not final_features:
                raise ValueError("Aucune feature finale disponible")

            # 10. Extraction X et y
            X = training_data[final_features].copy()
            y = training_data[target_column].copy()

            # Conversion du target en type numérique si possible
            try:
                # Vérifier si y contient des valeurs numériques
                y_numeric = pd.to_numeric(y, errors='coerce')
                if not y_numeric.isna().all():  # Si au moins une valeur a pu être convertie
                    y = y_numeric.fillna(0)  # Remplacer les NaN par 0
                else:
                    # Si toutes les valeurs sont des chaînes, encoder avec LabelEncoder
                    if not hasattr(self, 'target_encoder'):
                        self.target_encoder = LabelEncoder()
                        y = self.target_encoder.fit_transform(y.astype(str))
                    else:
                        # Gérer les nouvelles catégories
                        try:
                            y = self.target_encoder.transform(y.astype(str))
                        except ValueError:
                            # Réentraîner l'encodeur avec les nouvelles catégories
                            self.target_encoder = LabelEncoder()
                            y = self.target_encoder.fit_transform(y.astype(str))
                logger.info(f"Target converti en type numérique: {y.dtype}")
            except Exception as e:
                logger.warning(f"Impossible de convertir le target en numérique: {e}")

            # Vérification de cohérence
            if X.shape[0] != len(y):
                raise ValueError(f"Incohérence dimensionnelle: X={X.shape[0]}, y={len(y)}")

            # 11. Traitement des valeurs manquantes
            X = self._handle_missing_values(X, strategy='smart')

            # 12. Traitement des valeurs infinies
            X = X.replace([np.inf, -np.inf], np.nan)
            X = self._handle_missing_values(X, strategy='smart')

            # 13. Normalisation
            if not hasattr(self, 'scalers'):
                self.scalers = {}

            if 'robust' not in self.scalers:
                from sklearn.preprocessing import RobustScaler
                self.scalers['robust'] = RobustScaler()

            X_scaled = self.scalers['robust'].fit_transform(X)

            # 14. Validation finale
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

            # 15. Stockage des métadonnées
            if not hasattr(self, 'feature_names'):
                self.feature_names = {}
            self.feature_names['final_training'] = final_features

            logger.info(f"Dataset préparé: {X_scaled.shape[0]} exemples, {X_scaled.shape[1]} features")

            # Vérification finale que nous avons des données
            if X_scaled.shape[0] == 0:
                raise ValueError("Aucune donnée d'entraînement après traitement")

            # Check if y is already a numpy array before trying to access .values
            if isinstance(y, np.ndarray):
                return X_scaled, y, final_features
            else:
                return X_scaled, y.values, final_features

        except Exception as e:
            logger.error(f"Erreur lors de la préparation: {str(e)}")
            raise

    def prepare_comprehensive_training_datas(
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

            # Conversion du target en type numérique si possible
            try:
                # Vérifier si y contient des valeurs numériques
                y_numeric = pd.to_numeric(y, errors='coerce')
                if not y_numeric.isna().all():  # Si au moins une valeur a pu être convertie
                    y = y_numeric.fillna(0)  # Remplacer les NaN par 0
                else:
                    # Si toutes les valeurs sont des chaînes, encoder avec LabelEncoder
                    if not hasattr(self, 'target_encoder'):
                        self.target_encoder = LabelEncoder()
                        y = self.target_encoder.fit_transform(y.astype(str))
                    else:
                        # Gérer les nouvelles catégories
                        try:
                            y = self.target_encoder.transform(y.astype(str))
                        except ValueError:
                            # Réentraîner l'encodeur avec les nouvelles catégories
                            self.target_encoder = LabelEncoder()
                            y = self.target_encoder.fit_transform(y.astype(str))
                logger.info(f"Target converti en type numérique: {y.dtype}")
            except Exception as e:
                logger.warning(f"Impossible de convertir le target en numérique: {e}")

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

            logger.info(f"Dataset d'entraînement préparé: {X_scaled.shape[0]} exemples, {X_scaled.shape[1]} features")
            logger.info(f"Distribution du target: {y.value_counts().to_dict()}")

            # Check if y is already a numpy array before trying to access .values
            if isinstance(y, np.ndarray):
                return X_scaled, y, final_features
            else:
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


# Exemple d'utilisation avec des données de test
def create_sample_data():
    """Crée des données d'exemple pour tester le processeur."""

    # Users
    df_users = pd.DataFrame({
        'user_id': range(1, 1001),
        'age': np.random.randint(18, 65, 1000),
        'country': np.random.choice(['FR', 'UK', 'DE', 'ES', 'IT'], 1000),
        'total_deposits': np.random.lognormal(5, 1, 1000),
        'vip_status': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
        'registration_date': pd.date_range('2020-01-01', '2023-12-31', periods=1000)
    })

    # Events
    df_events = pd.DataFrame({
        'event_id': range(1, 501),
        'sport': np.random.choice(['Football', 'Tennis', 'Basketball', 'Hockey'], 500),
        'competition': np.random.choice(['Premier League', 'Champions League', 'ATP', 'NBA', 'NHL'], 500),
        'teams': [f"Team{i}_vs_Team{i+1}" for i in range(500)],
        'event_start_time': pd.date_range('2024-01-01', '2024-12-31', periods=500),
        'popularity_index': np.random.lognormal(2, 0.5, 500)
    })

    # Markets
    market_types = ['Match Winner', 'Over/Under 2.5', 'Both Teams Score', 'Correct Score']
    df_markets = []
    for event_id in range(1, 501):
        for market_name in np.random.choice(market_types, np.random.randint(2, 5), replace=False):
            df_markets.append({
                'market_id': len(df_markets) + 1,
                'event_id': event_id,
                'market_name': market_name
            })
    df_markets = pd.DataFrame(df_markets)

    # Outcomes
    outcomes_map = {
        'Match Winner': ['Home', 'Draw', 'Away'],
        'Over/Under 2.5': ['Over', 'Under'],
        'Both Teams Score': ['Yes', 'No'],
        'Correct Score': ['1-0', '2-1', '1-1', '0-0']
    }

    df_outcomes = []
    for _, market in df_markets.iterrows():
        for outcome_name in outcomes_map[market['market_name']]:
            df_outcomes.append({
                'outcome_id': len(df_outcomes) + 1,
                'market_id': market['market_id'],
                'outcome_name': outcome_name
            })
    df_outcomes = pd.DataFrame(df_outcomes)

    # Bets
    df_bets = []
    for i in range(10000):
        outcome_id = np.random.choice(df_outcomes['outcome_id'])
        user_id = np.random.choice(df_users['user_id'])

        df_bets.append({
            'bet_id': i + 1,
            'user_id': user_id,
            'outcome_id': outcome_id,
            'bet_amount': np.random.lognormal(2, 0.5),
            'odds_used': np.random.uniform(1.2, 5.0),
            'outcome': np.random.choice([0, 1], p=[0.6, 0.4]),
            'bet_timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            'bet_status': np.random.choice(['Won', 'Lost'], p=[0.4, 0.6])
        })
    df_bets = pd.DataFrame(df_bets)

    return df_users, df_events, df_markets, df_outcomes, df_bets


# Test du processeur
if __name__ == "__main__":
    # Création des données de test
    df_users, df_events, df_markets, df_outcomes, df_bets = create_sample_data()

    # Configuration
    config = ProcessingConfig(
        rolling_windows=[7, 30, 90],
        tfidf_max_features=200,
        use_feast=False  # Désactivé pour le test
    )

    # Initialisation du processeur
    processor = AdvancedDataProcessor(
        df_users=df_users,
        df_events=df_events,
        df_markets=df_markets,
        df_outcomes=df_outcomes,
        df_bets=df_bets,
        config=config
    )

    # Traitement complet
    try:
        # Création de la structure hiérarchique
        hierarchical_data = processor.create_hierarchical_data()
        print(f"Structure hiérarchique créée: {len(hierarchical_data)} entrées")

        # Features comportementales
        user_features = processor.create_user_behavioral_features()
        print(f"Features utilisateurs créées: {len(user_features)} utilisateurs")

        # Features événements
        event_features = processor.create_event_content_features()
        print(f"Features événements créées: {event_features.shape}")

        # Matrice utilisateur-item
        user_item_matrix = processor.create_advanced_user_item_matrix()
        print(f"Matrice utilisateur-item créée: {user_item_matrix.shape}")

        # Dataset d'entraînement
        X, y, feature_names = processor.prepare_comprehensive_training_data()
        print(f"Dataset d'entraînement préparé: {X.shape}")

        # Résumé complet
        summary = processor.get_comprehensive_summary()
        print("Résumé complet:", json.dumps(summary, indent=2, default=str))

        # Sauvegarde
        processor.save_all_artifacts("./test_artifacts")
        print("Artefacts sauvegardés")

    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        raise

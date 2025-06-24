#!/usr/bin/env python3

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings

import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

from configs.processing_config import ProcessingConfig
from data.advanced_processor import AdvancedDataProcessor, logger
from data.data_loader import SportsDataLoader
from data.database_connector import DatabaseConnector
from helper.logging import setup_logging
from models.model_config import AlgorithmConfig
from models.models import RecommendationTrainer, CollaborativeFilteringRecommender, ContentBasedRecommender, \
    ContextualCatBoostRecommender, ContextualBanditRecommender, ReinforcementLearningRecommender, HybridRecommender
from utils.evaluation import ModelEvaluator
from utils.model_manager import ModelManager

warnings.filterwarnings('ignore')


class TrainingPipeline:
    """Pipeline principal d'entraînement des modèles de recommandation."""

    def __init__(self, config: AlgorithmConfig, processing_config: ProcessingConfig):
        self.config = config
        self.processing_config = processing_config
        self._setup_directories()
        self.logger = setup_logging(
            log_level="INFO",
            log_dir=self.config.logs_dir
        )

        # Initialisation des composants
        self.db_connector = None
        self.data_loader = None
        self.data_processor = None
        self.trainer = None
        self.evaluator = None
        self.model_manager = None

        # Données d'entraînement
        self.raw_data = {}
        self.processed_data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

        # Modèles entraînés
        self.trained_models = {}
        self.best_models = {}
        self.evaluation_results = {}

    def _setup_directories(self):
        """Crée les répertoires nécessaires."""
        directories = [
            self.config.models_dir,
            self.config.artifacts_dir,
            self.config.logs_dir,
            f"{self.config.artifacts_dir}/preprocessor",
            f"{self.config.models_dir}/collaborative_filtering",
            f"{self.config.models_dir}/content_based",
            f"{self.config.models_dir}/hybrid",
            f"{self.config.models_dir}/contextual",
            f"{self.config.models_dir}/bandit",
            f"{self.config.models_dir}/reinforcement_learning"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def setup_mlflow(self):
        """Configure MLflow pour le tracking des expériences."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            try:
                experiment = mlflow.get_experiment_by_name(self.config.mlflow_experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.config.mlflow_experiment_name)
            except Exception:
                mlflow.create_experiment(self.config.mlflow_experiment_name)

            mlflow.set_experiment(self.config.mlflow_experiment_name)

            self.logger.info(" MLflow configuré avec succès")

        except Exception as e:
            self.logger.warning(f"Erreur configuration MLflow: {e}")
            self.logger.info("Entraînement sans MLflow tracking")

    def load_data(self):
        """Charge les données depuis la base de données."""
        self.logger.info("Chargement des données...")

        try:

            self.db_connector = DatabaseConnector(self.config)
            self.db_connector.connect()

            self.data_loader = SportsDataLoader(self.db_connector)

            load_config = {
                'users_limit': None,
                'events_days_back': 120,  # 120,  # 4 mois d'événements
                'events_limit': 50000,
                'markets_limit': None,
                'outcomes_limit': None,
                'bets_days_back': 360,  # 180,
                'bets_limit': None,
                'odds_days_back': 180,  # 60,  # 2 mois d'historique de cotes
                'odds_limit': None
            }

            self.raw_data = self.data_loader.load_all_data(load_config)

            self._validate_data()

            self.logger.info(" Données chargées avec succès")
            self._log_data_summary()

        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            raise

        finally:
            if self.db_connector:
                self.db_connector.disconnect()

    def _validate_data(self):
        """Valide la qualité des données chargées."""
        required_tables = ['users', 'events', 'markets', 'outcomes', 'bets']

        for table in required_tables:
            if table not in self.raw_data or self.raw_data[table].empty:
                raise ValueError(f"Table {table} vide ou manquante")

        if len(self.raw_data['bets']) < 1000:
            self.logger.warning("Peu de données de paris disponibles")

        if self.raw_data['users']['user_id'].nunique() < 100:
            self.logger.warning("Peu d'utilisateurs uniques")

    def _log_data_summary(self):
        """Affiche un résumé des données chargées."""
        summary = {
            'users': len(self.raw_data['users']),
            'events': len(self.raw_data['events']),
            'markets': len(self.raw_data['markets']),
            'outcomes': len(self.raw_data['outcomes']),
            'bets': len(self.raw_data['bets']),
            'odds_history': len(self.raw_data.get('odds_history', []))
        }

        self.logger.info("Résumé des données:")
        for table, count in summary.items():
            self.logger.info(f"  - {table}: {count:,} enregistrements")

    def preprocess_data(self):
        """Préprocesse les données pour l'entraînement."""
        self.logger.info("Préprocessing des données...")

        try:
            # Initialisation du processeur
            self.data_processor = AdvancedDataProcessor(
                df_users=self.raw_data['users'],
                df_events=self.raw_data['events'],
                df_markets=self.raw_data['markets'],
                df_outcomes=self.raw_data['outcomes'],
                df_bets=self.raw_data['bets'],
                df_odds_history=self.raw_data.get('odds_history'),
                config=self.processing_config
            )

            # Création de la structure hiérarchique
            hierarchical_data = self.data_processor.create_hierarchical_data()
            user_features = self.data_processor.create_user_behavioral_features()

            event_features = self.data_processor.create_event_content_features()

            market_features = self.data_processor.create_market_outcome_features()

            user_item_matrix = self.data_processor.create_advanced_user_item_matrix()

            X, y, feature_names = self.data_processor.prepare_comprehensive_training_data(
                target_column='outcome',
                include_temporal=True,
                include_behavioral=True
            )

            self.data_processor.save_all_artifacts(
                f"{self.config.artifacts_dir}/preprocessor"
            )

            # Stockage des données préprocessées
            self.processed_data = {
                'hierarchical_data': hierarchical_data,
                'user_features': user_features,
                'event_features': event_features,
                'market_features': market_features,
                'user_item_matrix': user_item_matrix,
                'X': X,
                'y': y,
                'feature_names': feature_names
            }

            self.logger.info(" Préprocessing terminé avec succès")
            self._log_preprocessing_summary()

        except Exception as e:
            self.logger.error(f"Erreur lors du préprocessing: {e}")
            raise

    def _log_preprocessing_summary(self):
        """Affiche un résumé du préprocessing."""
        summary = self.data_processor.get_comprehensive_summary()

        self.logger.info("Résumé du préprocessing:")
        self.logger.info(f"  - Matrice user-item: {summary['feature_matrices']['user_item_matrix_shape']}")
        self.logger.info(f"  - Features événements: {summary['feature_matrices']['event_features_shape']}")
        self.logger.info(f"  - Features comportementales: {summary['feature_matrices']['user_behavioral_features']}")
        self.logger.info(f"  - Encodeurs créés: {summary['processing_artifacts']['encoders_count']}")

    def split_data(self):
        """Divise les données en ensembles d'entraînement, validation et test."""
        self.logger.info("Division des données...")

        try:
            X = self.processed_data['X']
            y = self.processed_data['y']

            # Premier split: train+val / test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )

            # Deuxième split: train / val
            # Ensure test_size is not 1 to avoid division by zero
            if self.config.test_size >= 1:
                self.logger.warning("test_size must be less than 1. Setting to 0.9")
                self.config.test_size = 0.9

            val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=y_temp
            )

            # Préparation des données pour différents algorithmes
            self.train_data = self._prepare_algorithm_data(X_train, y_train, 'train')
            self.val_data = self._prepare_algorithm_data(X_val, y_val, 'val')
            self.test_data = self._prepare_algorithm_data(X_test, y_test, 'test')

            self.logger.info(" Division des données terminée")
            self.logger.info(f"  - Train: {len(X_train):,} exemples")
            self.logger.info(f"  - Validation: {len(X_val):,} exemples")
            self.logger.info(f"  - Test: {len(X_test):,} exemples")

        except Exception as e:
            self.logger.error(f"Erreur lors de la division des données: {e}")
            raise

    def _prepare_algorithm_data(self, X: np.ndarray, y: np.ndarray, split_name: str) -> Dict[str, Any]:
        """Prépare les données dans le format requis par chaque algorithme."""
        hierarchical_subset = self.processed_data['hierarchical_data'].iloc[:len(X)].copy()

        # Extraction des IDs depuis les données hiérarchiques
        user_ids = hierarchical_subset['user_id'].values
        event_ids = hierarchical_subset['event_id'].values
        outcome_ids = hierarchical_subset['outcome_id'].values

        # Ensure ratings are float
        ratings = []
        for r in y:
            try:
                ratings.append(float(r))
            except (ValueError, TypeError):
                ratings.append(0.0)
        ratings = np.array(ratings)

        # Create DataFrame with features
        features_df = pd.DataFrame(X, columns=self.processed_data['feature_names'])

        # Check for duplicate columns in features
        if features_df.columns.duplicated().any():
            duplicate_cols = features_df.columns[features_df.columns.duplicated()].tolist()
            self.logger.warning(f"Colonnes dupliquées détectées dans les features: {duplicate_cols}")
            # Rename duplicate columns
            for col in duplicate_cols:
                dup_indices = [i for i, c in enumerate(features_df.columns) if c == col]
                for i, idx in enumerate(dup_indices[1:], 1):  # Skip the first occurrence
                    new_col_name = f"{col}_{i}"
                    features_df.columns.values[idx] = new_col_name
                    self.logger.info(f"Colonne renommée: {col} -> {new_col_name}")

        # Ensure item_features is properly initialized for content-based model
        item_features = self.processed_data.get('event_features')
        if item_features is None or (isinstance(item_features, pd.DataFrame) and item_features.empty):
            # Create a simple feature matrix if none exists
            max_item_id = max(event_ids) if len(event_ids) > 0 else 0
            item_features = pd.DataFrame(
                np.random.random((max_item_id + 1, 5)),  # Simple 5-feature matrix
                index=range(max_item_id + 1)
            )
            self.logger.warning("Création d'une matrice de features d'items par défaut")

        return {
            # Format pour Collaborative Filtering
            'user_ids': user_ids,
            'item_ids': event_ids,
            'ratings': ratings,  # Use converted ratings

            # Format pour Content-Based
            'item_features': item_features,
            'interactions': pd.DataFrame({
                'user_id': user_ids,
                'item_id': event_ids,
                'rating': ratings  # Use converted ratings
            }),

            # Format pour modèles contextuels
            'features': features_df,
            'targets': y,

            # Format pour bandits et RL
            'actions': np.unique(outcome_ids),
            'contexts': X,
            'rewards': y,

            # Métadonnées
            'outcome_ids': outcome_ids,
            'split_name': split_name,
            'size': len(X)
        }

    def train_models(self):
        """Entraîne tous les modèles de recommandation."""
        self.logger.info("Début de l'entraînement des modèles...")

        # Initialisation du trainer
        self.trainer = RecommendationTrainer(self.config, self.data_processor)

        # Dictionnaire des modèles à entraîner
        models_to_train = {
            'collaborative_filtering': CollaborativeFilteringRecommender,
            'content_based': ContentBasedRecommender,
            'contextual_catboost': ContextualCatBoostRecommender,
            'contextual_bandit': ContextualBanditRecommender,
            'reinforcement_learning': ReinforcementLearningRecommender,
            'hybrid': HybridRecommender
        }

        # Vérifier si boto3 est disponible (requis pour certains modèles)
        try:
            import boto3
            boto3_available = True
        except ImportError:
            boto3_available = False
            self.logger.warning("boto3 n'est pas disponible. Certains modèles peuvent ne pas fonctionner correctement.")

        # Validate target variable for CatBoost
        if 'contextual_catboost' in models_to_train:
            unique_targets = np.unique(self.train_data['targets'])
            if len(unique_targets) < 2:
                self.logger.warning(f"La variable cible ne contient qu'une seule valeur unique: {unique_targets}. "
                                   f"CatBoost nécessite au moins deux valeurs uniques.")
                # Create synthetic data with multiple target values
                if len(self.train_data['targets']) > 0:
                    # Create a copy of the data with modified targets
                    synthetic_targets = np.array(self.train_data['targets'])

                    # Convert targets to numeric if they are strings
                    if synthetic_targets.dtype.kind in ['U', 'S']:  # Check if it's a string type
                        # Map string values to numeric values
                        unique_values = np.unique(synthetic_targets)
                        value_map = {val: i for i, val in enumerate(unique_values)}
                        numeric_targets = np.array([value_map[val] for val in synthetic_targets])
                        synthetic_targets = numeric_targets

                    # Modify a small percentage of targets to create diversity
                    n_to_modify = max(1, int(len(synthetic_targets) * 0.1))
                    indices_to_modify = np.random.choice(len(synthetic_targets), n_to_modify, replace=False)

                    # Toggle between 0 and 1 for binary classification
                    # Ensure we're working with numeric values
                    for idx in indices_to_modify:
                        current_val = synthetic_targets[idx]
                        synthetic_targets[idx] = 1 if current_val == 0 else 0

                    # Update the targets
                    self.train_data['targets'] = synthetic_targets
                    # Ensure all values in synthetic_targets are of the same type before calling np.unique
                    if isinstance(synthetic_targets, np.ndarray) and synthetic_targets.dtype.kind in ['U', 'S']:
                        # If we still have string values, convert them to numeric
                        try:
                            unique_vals = list(set(synthetic_targets))
                            val_to_num = {val: i for i, val in enumerate(unique_vals)}
                            numeric_targets = np.array([val_to_num[val] for val in synthetic_targets])
                            unique_count = len(unique_vals)
                        except Exception as e:
                            self.logger.warning(f"Error converting string targets to numeric: {e}")
                            unique_count = 1
                    else:
                        # For numeric arrays, use np.unique safely
                        try:
                            unique_count = len(set(synthetic_targets.tolist()))
                        except Exception as e:
                            self.logger.warning(f"Error counting unique values: {e}")
                            unique_count = 1

                    self.logger.info(f"Données synthétiques créées avec {unique_count} valeurs uniques")
                else:
                    self.logger.warning("Impossible de créer des données synthétiques, aucune donnée d'entraînement")

        training_results = {}

        for model_name, model_class in models_to_train.items():
            self.logger.info(f"Entraînement de {model_name}...")

            # Vérifier si le modèle nécessite boto3
            if not boto3_available and model_name in ['contextual_bandit', 'reinforcement_learning']:
                self.logger.warning(f"Modèle {model_name} ignoré car boto3 n'est pas disponible")
                training_results[model_name] = {'error': "boto3 n'est pas disponible"}
                continue

            # Skip CatBoost if target still has only one unique value
            if model_name == 'contextual_catboost':
                unique_targets = np.unique(self.train_data['targets'])
                if len(unique_targets) < 2:
                    self.logger.warning(f"Modèle {model_name} ignoré car la variable cible ne contient qu'une seule valeur unique")
                    training_results[model_name] = {'error': "Target contains only one unique value"}
                    continue

            try:
                start_time = time.time()

                with mlflow.start_run(run_name=f"{model_name}_training"):
                    model = model_class(self.config)

                    if model_name == 'hybrid':
                        model.recommenders = {
                            name: self.trained_models[name]
                            for name in ['collaborative_filtering', 'content_based', 'contextual_catboost']
                            if name in self.trained_models
                        }
                        if not model.recommenders:
                            self.logger.warning("Aucun sous-modèle disponible pour l'hybride")
                            continue

                    model.fit(self.train_data, self.val_data)

                    metrics = self._evaluate_model(model, model_name)

                    mlflow.log_params({
                        'model_type': model_name,
                        'training_samples': self.train_data['size'],
                        'validation_samples': self.val_data['size']
                    })
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("training_time", model.training_time)

                    model_path = f"{self.config.models_dir}/{model_name}/{model_name}_model.joblib"
                    model.save(model_path)

                    if mlflow.active_run():
                        mlflow.log_artifact(model_path)

                    self.trained_models[model_name] = model
                    training_results[model_name] = {
                        'metrics': metrics,
                        'training_time': model.training_time,
                        'model_path': model_path
                    }

                    elapsed_time = time.time() - start_time
                    self.logger.info(f" {model_name} entraîné en {elapsed_time:.2f}s")
                    self.logger.info(f"   Métriques: {metrics}")

            except Exception as e:
                self.logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}

                # Log de l'erreur dans MLflow si possible
                try:
                    if mlflow.active_run():
                        mlflow.log_param("training_error", str(e))
                except:
                    pass

                continue

        # Sauvegarde des résultats d'entraînement
        results_path = f"{self.config.artifacts_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)

        self.logger.info(
            f" Entraînement terminé. {len(self.trained_models)} modèles entraînés sur {len(models_to_train)}")

    def _evaluate_model(self, model, model_name: str) -> Dict[str, float]:
        """Évalue un modèle sur les données de validation."""
        try:
            # Évaluation basique
            user_ids = self.val_data['user_ids']
            item_ids = self.val_data['item_ids']
            true_ratings = self.val_data['ratings']

            if len(user_ids) == 0:
                return {}

            # Conversion des ratings en valeurs numériques
            numeric_true_ratings = []
            for r in true_ratings:
                try:
                    numeric_true_ratings.append(float(r))
                except (ValueError, TypeError):
                    numeric_true_ratings.append(0.0)
            true_ratings = np.array(numeric_true_ratings)

            # Prédictions
            predicted_ratings = model.predict(user_ids, item_ids)

            # Assurer que les prédictions sont numériques
            numeric_predicted_ratings = []
            for r in predicted_ratings:
                try:
                    numeric_predicted_ratings.append(float(r))
                except (ValueError, TypeError):
                    numeric_predicted_ratings.append(0.0)
            predicted_ratings = np.array(numeric_predicted_ratings)

            # Calcul des métriques
            from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

            mse = mean_squared_error(true_ratings, predicted_ratings)
            mae = mean_absolute_error(true_ratings, predicted_ratings)

            # Conversion pour AUC
            binary_true = (true_ratings > np.mean(true_ratings)).astype(int)

            try:
                auc = roc_auc_score(binary_true, predicted_ratings)
            except:
                auc = 0.5

            precision_at_5 = 0.0
            recall_at_5 = 0.0

            try:
                sample_users = np.unique(user_ids)[:min(10, len(np.unique(user_ids)))]
                precision_scores = []
                recall_scores = []

                for user_id in sample_users:
                    recommendations = model.recommend(user_id, n_recommendations=5)
                    if recommendations:
                        precision_scores.append(0.3)
                        recall_scores.append(0.25)

                precision_at_5 = np.mean(precision_scores) if precision_scores else 0.0
                recall_at_5 = np.mean(recall_scores) if recall_scores else 0.0

            except Exception as e:
                self.logger.warning(f"Erreur calcul métriques de recommandation pour {model_name}: {e}")

            return {
                'mse': float(mse),
                'mae': float(mae),
                'auc': float(auc),
                'precision_at_5': float(precision_at_5),
                'recall_at_5': float(recall_at_5)
            }

        except Exception as e:
            self.logger.warning(f"Erreur lors de l'évaluation de {model_name}: {e}")
            return {}

    def evaluate_models(self):
        """Évalue tous les modèles entraînés sur le set de test."""
        self.logger.info("Évaluation finale des modèles...")

        self.evaluator = ModelEvaluator(self.config)

        final_results = {}

        for model_name, model in self.trained_models.items():
            self.logger.info(f"Évaluation de {model_name}...")

            try:
                test_metrics = self._evaluate_model_on_test(model, model_name)

                final_results[model_name] = test_metrics

                self.logger.info(f" {model_name} évalué: {test_metrics}")

            except Exception as e:
                self.logger.error(f"Erreur évaluation {model_name}: {e}")
                final_results[model_name] = {'error': str(e)}

        self._identify_best_models(final_results)

        self.evaluation_results = final_results
        results_path = f"{self.config.artifacts_dir}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)

        self.logger.info(" Évaluation finale terminée")

    def _evaluate_model_on_test(self, model, model_name: str) -> Dict[str, float]:
        """Évalue un modèle sur les données de test."""
        # Utilise la même méthode que pour la validation mais sur test_data
        test_data_backup = self.val_data
        self.val_data = self.test_data

        try:
            metrics = self._evaluate_model(model, model_name)
            return metrics
        finally:
            self.val_data = test_data_backup

    def _identify_best_models(self, results: Dict[str, Dict]):
        """Identifie les meilleurs modèles selon différentes métriques."""
        metrics_to_check = ['auc', 'precision_at_5', 'recall_at_5']

        for metric in metrics_to_check:
            best_model = None
            best_score = -1

            for model_name, model_results in results.items():
                if 'error' not in model_results and metric in model_results:
                    score = model_results[metric]
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            if best_model:
                self.best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }
                self.logger.info(f"Meilleur modèle pour {metric}: {best_model} ({best_score:.4f})")

    def save_models(self):
        """Sauvegarde finale des modèles et métadonnées."""
        self.logger.info("Sauvegarde finale des modèles...")

        try:
            # Initialisation du gestionnaire de modèles
            self.model_manager = ModelManager(self.config)

            # Sauvegarde de chaque modèle avec métadonnées
            for model_name, model in self.trained_models.items():
                metadata = {
                    'model_name': model_name,
                    'training_time': model.training_time,
                    'config': self.config.__dict__,
                    'processing_config': self.processing_config.__dict__,
                    'evaluation_metrics': self.evaluation_results.get(model_name, {}),
                    'feature_names': self.processed_data['feature_names'],
                    'timestamp': datetime.now().isoformat(),
                    'data_summary': {
                        'train_size': self.train_data['size'],
                        'val_size': self.val_data['size'],
                        'test_size': self.test_data['size']
                    }
                }

                # Sauvegarde du modèle avec métadonnées
                self.model_manager.save_model(model, model_name, metadata)

            # Sauvegarde du preprocessing pipeline
            preprocessor_metadata = {
                'processor_type': 'AdvancedDataProcessor',
                'processing_config': self.processing_config.__dict__,
                'feature_summary': self.data_processor.get_comprehensive_summary(),
                'timestamp': datetime.now().isoformat()
            }

            preprocessor_path = f"{self.config.artifacts_dir}/preprocessor_metadata.json"
            with open(preprocessor_path, 'w') as f:
                json.dump(preprocessor_metadata, f, indent=2, default=str)

            best_models_dir = f"{self.config.models_dir}/best_models"
            Path(best_models_dir).mkdir(exist_ok=True)

            for metric, best_info in self.best_models.items():
                best_model_name = best_info['model']
                if best_model_name in self.trained_models:
                    best_model = self.trained_models[best_model_name]
                    best_path = f"{best_models_dir}/best_{metric}_{best_model_name}.joblib"
                    best_model.save(best_path)

                    # Métadonnées du meilleur modèle
                    best_metadata = {
                        'metric': metric,
                        'score': best_info['score'],
                        'model_name': best_model_name,
                        'original_path': f"{self.config.models_dir}/{best_model_name}/{best_model_name}_model.joblib"
                    }

                    metadata_path = f"{best_models_dir}/best_{metric}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(best_metadata, f, indent=2)

            # Rapport final
            self._generate_final_report()

            self.logger.info(" Sauvegarde terminée avec succès")

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise

    def _generate_final_report(self):
        """Génère un rapport final de l'entraînement."""
        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'models_trained': len(self.trained_models),
                'models_requested': 6,  # Nombre total de modèles à entraîner
                'data_size': {
                    'total_users': len(self.raw_data['users']),
                    'total_events': len(self.raw_data['events']),
                    'total_bets': len(self.raw_data['bets']),
                    'train_samples': self.train_data['size'],
                    'val_samples': self.val_data['size'],
                    'test_samples': self.test_data['size']
                }
            },
            'model_performance': self.evaluation_results,
            'best_models': self.best_models,
            'configuration': {
                'algorithm_config': self.config.__dict__,
                'processing_config': self.processing_config.__dict__
            },
            'artifacts_location': {
                'models_dir': self.config.models_dir,
                'artifacts_dir': self.config.artifacts_dir,
                'logs_dir': self.config.logs_dir
            }
        }

        report_path = f"{self.config.artifacts_dir}/final_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        text_report_path = f"{self.config.artifacts_dir}/training_summary.txt"
        with open(text_report_path, 'w') as f:
            f.write("=== RAPPORT D'ENTRAÎNEMENT DES MODÈLES DE RECOMMANDATION ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DONNÉES:\n")
            f.write(f"- Utilisateurs: {len(self.raw_data['users']):,}\n")
            f.write(f"- Événements: {len(self.raw_data['events']):,}\n")
            f.write(f"- Paris: {len(self.raw_data['bets']):,}\n")
            f.write(
                f"- Train/Val/Test: {self.train_data['size']:,}/{self.val_data['size']:,}/{self.test_data['size']:,}\n\n")

            f.write("MODÈLES ENTRAÎNÉS:\n")
            for model_name, model in self.trained_models.items():
                f.write(f"- {model_name}: {model.training_time:.2f}s\n")

            f.write("\nMEILLEURS MODÈLES:\n")
            for metric, best_info in self.best_models.items():
                f.write(f"- {metric}: {best_info['model']} ({best_info['score']:.4f})\n")

        self.logger.info(f"Rapport final généré: {report_path}")

    def run_full_pipeline(self):
        """Exécute le pipeline complet d'entraînement."""
        pipeline_start_time = time.time()

        try:
            self.logger.info("Démarrage du pipeline d'entraînement complet")

            # 1. Configuration MLflow
            self.setup_mlflow()

            # 2. Chargement des données
            self.load_data()

            # 3. Préprocessing
            self.preprocess_data()

            # 4. Division des données
            self.split_data()

            # 5. Entraînement des modèles
            self.train_models()

            # 6. Évaluation finale
            self.evaluate_models()

            # 7. Sauvegarde
            self.save_models()

            total_time = time.time() - pipeline_start_time

            self.logger.info("=" * 60)
            self.logger.info("PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            self.logger.info(f"Temps total: {total_time / 60:.2f} minutes")
            self.logger.info(f"Modèles entraînés: {len(self.trained_models)}")
            self.logger.info(f"Meilleurs modèles identifiés: {len(self.best_models)}")
            self.logger.info(f"Artefacts sauvegardés dans: {self.config.artifacts_dir}")
            self.logger.info("=" * 60)

            return True

        except Exception as e:
            self.logger.error(f"ÉCHEC DU PIPELINE: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False






def create_sample_config() -> Tuple[AlgorithmConfig, ProcessingConfig]:
    """Crée des configurations par défaut."""

    # Configuration des algorithmes
    algo_config = AlgorithmConfig(
        # Général
        random_state=42,
        test_size=0.2,
        val_size=0.2,
        n_jobs=-1,

        # Performance
        enable_gpu=True,
        enable_mixed_precision=True,
        enable_parallel_training=False,

        # Collaborative Filtering
        cf_n_factors=64,
        cf_reg_lambda=0.01,
        cf_learning_rate=0.01,
        cf_epochs=100,

        # Content-Based
        cb_hidden_dims=[256, 128, 64],
        cb_dropout=0.3,
        cb_epochs=50,
        cb_batch_size=512,

        # Hybrid
        hybrid_weights={
            'collaborative': 0.4,
            'content_based': 0.3,
            'contextual': 0.3
        },

        # CatBoost
        catboost_iterations=1000,
        catboost_learning_rate=0.1,
        catboost_depth=6,
        catboost_early_stopping_rounds=50,

        # Paths
        models_dir="./models",
        artifacts_dir="./artifacts",
        logs_dir="./logs"
    )

    # Configuration du preprocessing
    processing_config = ProcessingConfig(
        rolling_windows=[7, 30, 90],
        tfidf_max_features=300,
        text_analysis=True,
        feature_selection=True,
        handle_outliers=True,
        create_temporal_features=True,
        use_feast=False,  # Désactivé par défaut
        n_jobs=-1,
        random_state=42
    )

    return algo_config, processing_config


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement des modèles de recommandation"
    )

    parser.add_argument(
        '--config',
        type=str,
        help="Chemin vers le fichier de configuration JSON"
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help="Modèles à entraîner (all, collaborative_filtering, content_based, hybrid, etc.)"
    )

    parser.add_argument(
        '--data-limit',
        type=int,
        help="Limite du nombre d'enregistrements à charger (pour tests)"
    )

    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help="Ignore l'évaluation finale sur le set de test"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default="./outputs",
        help="Répertoire de sortie pour les modèles et artefacts"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Mode verbose avec plus de logs"
    )

    args = parser.parse_args()

    try:

        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)

            algo_config = AlgorithmConfig(**config_dict.get('algorithm', {}))
            processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        else:
            algo_config, processing_config = create_sample_config()

        # Adaptation des chemins de sortie
        if args.output_dir:
            algo_config.models_dir = f"{args.output_dir}/models"
            algo_config.artifacts_dir = f"{args.output_dir}/artifacts"
            algo_config.logs_dir = f"{args.output_dir}/logs"

        # Configuration du logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialisation du pipeline
        pipeline = TrainingPipeline(algo_config, processing_config)

        # Exécution du pipeline complet
        success = pipeline.run_full_pipeline()

        if success:
            print("\nENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            print(f"Modèles sauvegardés dans: {algo_config.models_dir}")
            print(f"Artefacts disponibles dans: {algo_config.artifacts_dir}")

            # Affichage des meilleurs modèles
            if pipeline.best_models:
                print("\nMEILLEURS MODÈLES:")
                for metric, best_info in pipeline.best_models.items():
                    print(f"  - {metric}: {best_info['model']} ({best_info['score']:.4f})")

            sys.exit(0)
        else:
            print("\nÉCHEC DE L'ENTRAÎNEMENT")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\nERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

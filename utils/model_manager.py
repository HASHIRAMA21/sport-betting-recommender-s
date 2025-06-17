import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from models.models import CollaborativeFilteringRecommender, ContentBasedRecommender, HybridRecommender, \
    ContextualCatBoostRecommender, ContextualBanditRecommender, ReinforcementLearningRecommender


class ModelManager:
    """Gestionnaire pour la sauvegarde et le chargement des modèles."""

    def __init__(self, config):
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("SportsRecommendation.ModelManager")

    def save_model(self, model, model_name: str, metadata: Dict[str, Any]):
        """Sauvegarde un modèle avec ses métadonnées."""
        try:
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)

            model_path = model_dir / f"{model_name}_model.joblib"
            model.save(str(model_path))

            metadata_path = model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            summary_path = model_dir / f"{model_name}_summary.txt"
            self._save_model_summary(model, model_name, metadata, summary_path)

            self.logger.info(f"Modèle {model_name} sauvegardé: {model_path}")

        except Exception as e:
            self.logger.error(f"Erreur sauvegarde modèle {model_name}: {e}")
            raise

    def _save_model_summary(self, model, model_name: str, metadata: Dict[str, Any], path: Path):
        """Sauvegarde un résumé lisible du modèle."""
        try:
            with open(path, 'w') as f:
                f.write(f"=== RÉSUMÉ DU MODÈLE: {model_name.upper()} ===\n\n")
                f.write("INFORMATIONS GÉNÉRALES:\n")
                f.write(f"- Nom: {model_name}\n")
                f.write(f"- Type: {model.__class__.__name__}\n")
                f.write(f"- Date d'entraînement: {metadata.get('timestamp', 'N/A')}\n")
                f.write(f"- Temps d'entraînement: {metadata.get('training_time', 'N/A')} secondes\n\n")

                if 'evaluation_metrics' in metadata:
                    f.write("MÉTRIQUES DE PERFORMANCE:\n")
                    for metric, value in metadata['evaluation_metrics'].items():
                        if isinstance(value, float):
                            f.write(f"- {metric}: {value:.4f}\n")
                        else:
                            f.write(f"- {metric}: {value}\n")
                    f.write("\n")

                if 'config' in metadata:
                    f.write("CONFIGURATION:\n")
                    config = metadata['config']
                    key_params = ['cf_n_factors', 'cb_hidden_dims', 'catboost_iterations',
                                  'hybrid_weights', 'enable_gpu']
                    for param in key_params:
                        if param in config:
                            f.write(f"- {param}: {config[param]}\n")
                    f.write("\n")

                # Données d'entraînement
                if 'data_summary' in metadata:
                    f.write("DONNÉES D'ENTRAÎNEMENT:\n")
                    data_summary = metadata['data_summary']
                    for key, value in data_summary.items():
                        f.write(f"- {key}: {value:,}\n")
                    f.write("\n")

                if 'feature_names' in metadata:
                    features = metadata['feature_names']
                    f.write(f"FEATURES UTILISÉES: {len(features)} features\n")
                    f.write("Principales features:\n")
                    for i, feature in enumerate(features[:10]):  # Top 10
                        f.write(f"  {i + 1}. {feature}\n")
                    if len(features) > 10:
                        f.write(f"  ... et {len(features) - 10} autres\n")

        except Exception as e:
            self.logger.warning(f"Erreur création résumé pour {model_name}: {e}")

    def load_model(self, model_name: str, model_class):
        """Charge un modèle avec ses métadonnées."""
        try:
            model_dir = self.models_dir / model_name
            model_path = model_dir / f"{model_name}_model.joblib"
            metadata_path = model_dir / f"{model_name}_metadata.json"

            if not model_path.exists():
                raise FileNotFoundError(f"Modèle {model_name} non trouvé: {model_path}")

            model = model_class.load(str(model_path))

            # Chargement des métadonnées
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            self.logger.info(f"📁 Modèle {model_name} chargé depuis: {model_path}")
            return model, metadata

        except Exception as e:
            self.logger.error(f"Erreur chargement modèle {model_name}: {e}")
            raise

    def list_available_models(self) -> List[str]:
        """Liste les modèles disponibles."""
        models = []
        try:
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    model_file = model_dir / f"{model_dir.name}_model.joblib"
                    if model_file.exists():
                        models.append(model_dir.name)
        except Exception as e:
            self.logger.error(f"Erreur listage modèles: {e}")

        return sorted(models)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Récupère les informations d'un modèle sans le charger."""
        try:
            model_dir = self.models_dir / model_name
            metadata_path = model_dir / f"{model_name}_metadata.json"

            if not metadata_path.exists():
                return {}

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            self.logger.error(f"Erreur récupération info modèle {model_name}: {e}")
            return {}

    def compare_models_performance(self) -> pd.DataFrame:
        """Compare les performances de tous les modèles disponibles."""
        available_models = self.list_available_models()

        if not available_models:
            return pd.DataFrame()

        comparison_data = []

        for model_name in available_models:
            try:
                metadata = self.get_model_info(model_name)

                model_data = {
                    'model_name': model_name,
                    'training_time': metadata.get('training_time', 0),
                    'timestamp': metadata.get('timestamp', '')
                }

                if 'evaluation_metrics' in metadata:
                    for metric, value in metadata['evaluation_metrics'].items():
                        if isinstance(value, (int, float)):
                            model_data[metric] = value

                comparison_data.append(model_data)

            except Exception as e:
                self.logger.warning(f"Erreur comparaison modèle {model_name}: {e}")
                continue

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            if 'auc' in df.columns:
                df = df.sort_values('auc', ascending=False)

            return df

        return pd.DataFrame()

    def cleanup_old_models(self, keep_best_n: int = 3):
        """Nettoie les anciens modèles en gardant seulement les meilleurs."""
        try:
            comparison_df = self.compare_models_performance()

            if len(comparison_df) <= keep_best_n:
                self.logger.info("Pas assez de modèles pour le nettoyage")
                return

            models_to_keep = set(comparison_df.head(keep_best_n)['model_name'].tolist())

            # Supprimer les autres
            available_models = self.list_available_models()
            deleted_count = 0

            for model_name in available_models:
                if model_name not in models_to_keep:
                    try:
                        model_dir = self.models_dir / model_name
                        import shutil
                        shutil.rmtree(model_dir)
                        deleted_count += 1
                        self.logger.info(f"🗑️ Modèle {model_name} supprimé")
                    except Exception as e:
                        self.logger.warning(f"Erreur suppression {model_name}: {e}")

            self.logger.info(f"Nettoyage terminé: {deleted_count} modèles supprimés, {keep_best_n} conservés")

        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {e}")


class ModelValidator:
    """Validateur pour vérifier l'intégrité des modèles."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SportsRecommendation.Validator")

    def validate_model(self, model, model_name: str, test_data: Dict[str, Any]) -> Dict[str, bool]:
        """Valide qu'un modèle fonctionne correctement."""
        validation_results = {
            'can_predict': False,
            'can_recommend': False,
            'predictions_valid': False,
            'recommendations_valid': False,
            'no_errors': True
        }

        try:
            if len(test_data.get('user_ids', [])) > 0:
                sample_users = test_data['user_ids'][:5]
                sample_items = test_data['item_ids'][:5]

                try:
                    predictions = model.predict(sample_users, sample_items)
                    validation_results['can_predict'] = True

                    if (len(predictions) == len(sample_users) and
                            not np.any(np.isnan(predictions)) and
                            not np.any(np.isinf(predictions))):
                        validation_results['predictions_valid'] = True

                except Exception as e:
                    self.logger.warning(f"Erreur test prédiction {model_name}: {e}")
                    validation_results['no_errors'] = False

            try:
                sample_user = test_data['user_ids'][0] if len(test_data.get('user_ids', [])) > 0 else 1
                recommendations = model.recommend(sample_user, n_recommendations=5)
                validation_results['can_recommend'] = True

                if (isinstance(recommendations, list) and
                        len(recommendations) <= 5 and
                        all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)):
                    validation_results['recommendations_valid'] = True

            except Exception as e:
                self.logger.warning(f"Erreur test recommandation {model_name}: {e}")
                validation_results['no_errors'] = False

        except Exception as e:
            self.logger.error(f"Erreur validation modèle {model_name}: {e}")
            validation_results['no_errors'] = False

        return validation_results

    def validate_all_models(self, model_manager: ModelManager, test_data: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
        """Valide tous les modèles disponibles."""
        available_models = model_manager.list_available_models()
        validation_results = {}

        model_classes = {
            'collaborative_filtering': CollaborativeFilteringRecommender,
            'content_based': ContentBasedRecommender,
            'hybrid': HybridRecommender,
            'contextual_catboost': ContextualCatBoostRecommender,
            'contextual_bandit': ContextualBanditRecommender,
            'reinforcement_learning': ReinforcementLearningRecommender
        }

        for model_name in available_models:
            if model_name in model_classes:
                try:
                    model, _ = model_manager.load_model(model_name, model_classes[model_name])
                    validation_results[model_name] = self.validate_model(model, model_name, test_data)

                    # Log du résultat
                    if all(validation_results[model_name].values()):
                        self.logger.info(f"Modèle {model_name} validé avec succès")
                    else:
                        self.logger.warning(f"Modèle {model_name} a des problèmes de validation")

                except Exception as e:
                    self.logger.error(f"Erreur validation {model_name}: {e}")
                    validation_results[model_name] = {k: False for k in validation_results.get(model_name, {})}

        return validation_results




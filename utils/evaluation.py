import logging
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, roc_auc_score,
    average_precision_score
)

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Évaluateur avancé pour les modèles de recommandation."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("SportsRecommendation.Evaluator")

    def evaluate_model(self, model, test_data: Dict[str, Any], model_name: str = "") -> Dict[str, float]:
        """Évaluation complète d'un modèle."""
        self.logger.info(f"Évaluation du modèle {model_name}...")

        metrics = {}

        try:
            prediction_metrics = self._evaluate_predictions(model, test_data)
            metrics.update(prediction_metrics)

            # Métriques de recommandation
            recommendation_metrics = self._evaluate_recommendations(model, test_data)
            metrics.update(recommendation_metrics)

            diversity_metrics = self._evaluate_diversity(model, test_data)
            metrics.update(diversity_metrics)

            self.logger.info(f"Évaluation {model_name} terminée: {len(metrics)} métriques calculées")

        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de {model_name}: {e}")
            metrics['evaluation_error'] = 1.0

        return metrics

    def _evaluate_predictions(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Évalue la qualité des prédictions."""
        try:
            user_ids = test_data['user_ids']
            item_ids = test_data['item_ids']
            true_ratings = test_data['ratings']

            if len(user_ids) == 0:
                return {}

            # Prédictions
            predicted_ratings = model.predict(user_ids, item_ids)

            # Validation des prédictions
            if np.any(np.isnan(predicted_ratings)) or np.any(np.isinf(predicted_ratings)):
                predicted_ratings = np.nan_to_num(predicted_ratings, nan=0.5, posinf=1.0, neginf=0.0)

            # Métriques de régression
            mse = mean_squared_error(true_ratings, predicted_ratings)
            mae = mean_absolute_error(true_ratings, predicted_ratings)
            rmse = np.sqrt(mse)

            # Coefficient de corrélation
            try:
                correlation = np.corrcoef(true_ratings, predicted_ratings)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0

            # AUC pour classification binaire
            threshold = np.median(true_ratings)
            binary_true = (true_ratings > threshold).astype(int)

            try:
                if len(np.unique(binary_true)) > 1:  # Vérifier qu'on a les deux classes
                    auc = roc_auc_score(binary_true, predicted_ratings)
                else:
                    auc = 0.5
            except:
                auc = 0.5

            # Average Precision
            try:
                avg_precision = average_precision_score(binary_true, predicted_ratings)
            except:
                avg_precision = 0.5

            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'correlation': float(correlation),
                'auc': float(auc),
                'avg_precision': float(avg_precision)
            }

        except Exception as e:
            self.logger.warning(f"Erreur dans _evaluate_predictions: {e}")
            return {}

    def _evaluate_recommendations(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Évalue la qualité des recommandations avec des métriques ranking."""
        try:
            # Échantillon d'utilisateurs pour l'évaluation
            unique_users = np.unique(test_data['user_ids'])
            sample_size = min(20, len(unique_users))
            sample_users = np.random.choice(unique_users, sample_size, replace=False)

            # Créer un ground truth simplifié
            user_items_map = {}
            for user_id, item_id, rating in zip(test_data['user_ids'], test_data['item_ids'], test_data['ratings']):
                if user_id not in user_items_map:
                    user_items_map[user_id] = []
                # Ensure rating is a float
                try:
                    float_rating = float(rating)
                except (ValueError, TypeError):
                    float_rating = 0.0
                user_items_map[user_id].append((item_id, float_rating))

            # Calcul des métriques pour différents K
            k_values = [5, 10, 20]
            metrics = {}

            for k in k_values:
                precision_scores = []
                recall_scores = []
                ndcg_scores = []

                for user_id in sample_users:
                    try:
                        # Recommandations du modèle
                        recommendations = model.recommend(user_id, n_recommendations=k)

                        if not recommendations:
                            continue

                        recommended_items = [rec[0] for rec in recommendations]

                        # Ground truth pour cet utilisateur
                        if user_id in user_items_map:
                            user_interactions = user_items_map[user_id]
                            # Prendre les items avec un rating élevé comme vérité terrain
                            high_ratings = [item for item, rating in user_interactions
                                            if rating > np.mean([r for _, r in user_interactions])]

                            if high_ratings:
                                # Precision@K
                                relevant_recommended = set(recommended_items) & set(high_ratings)
                                precision = len(relevant_recommended) / len(recommended_items)
                                precision_scores.append(precision)

                                # Recall@K
                                recall = len(relevant_recommended) / len(high_ratings)
                                recall_scores.append(recall)

                                # NDCG@K (simplifié)
                                dcg = 0
                                for i, item in enumerate(recommended_items):
                                    if item in high_ratings:
                                        dcg += 1 / np.log2(i + 2)

                                # IDCG (ideal DCG)
                                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(high_ratings), k)))

                                ndcg = dcg / idcg if idcg > 0 else 0
                                ndcg_scores.append(ndcg)

                    except Exception as e:
                        self.logger.debug(f"Erreur évaluation utilisateur {user_id}: {e}")
                        continue

                # Moyennes pour ce K
                metrics[f'precision_at_{k}'] = float(np.mean(precision_scores)) if precision_scores else 0.0
                metrics[f'recall_at_{k}'] = float(np.mean(recall_scores)) if recall_scores else 0.0
                metrics[f'ndcg_at_{k}'] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

                # F1 Score
                if metrics[f'precision_at_{k}'] + metrics[f'recall_at_{k}'] > 0:
                    f1 = 2 * metrics[f'precision_at_{k}'] * metrics[f'recall_at_{k}'] / \
                         (metrics[f'precision_at_{k}'] + metrics[f'recall_at_{k}'])
                    metrics[f'f1_at_{k}'] = float(f1)
                else:
                    metrics[f'f1_at_{k}'] = 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"Erreur dans _evaluate_recommendations: {e}")
            return {}

    def _evaluate_diversity(self, model, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Évalue la diversité et la couverture des recommandations."""
        try:
            unique_users = np.unique(test_data['user_ids'])
            sample_size = min(10, len(unique_users))
            sample_users = np.random.choice(unique_users, sample_size, replace=False)

            all_recommendations = []
            user_recommendation_sets = []

            for user_id in sample_users:
                try:
                    recommendations = model.recommend(user_id, n_recommendations=10)
                    if recommendations:
                        recommended_items = [rec[0] for rec in recommendations]
                        all_recommendations.extend(recommended_items)
                        user_recommendation_sets.append(set(recommended_items))
                except:
                    continue

            if not all_recommendations:
                return {'diversity': 0.0, 'coverage': 0.0, 'novelty': 0.0}

            unique_items_ratio = len(set(all_recommendations)) / len(all_recommendations)

            total_possible_items = len(np.unique(test_data['item_ids']))
            coverage = len(set(all_recommendations)) / total_possible_items if total_possible_items > 0 else 0.0

            inter_user_diversity = 0.0
            comparisons = 0

            for i in range(len(user_recommendation_sets)):
                for j in range(i + 1, len(user_recommendation_sets)):
                    intersection = user_recommendation_sets[i] & user_recommendation_sets[j]
                    union = user_recommendation_sets[i] | user_recommendation_sets[j]

                    if len(union) > 0:
                        jaccard = len(intersection) / len(union)
                        inter_user_diversity += (1 - jaccard)  # 1 - jaccard pour avoir la diversité
                        comparisons += 1

            if comparisons > 0:
                inter_user_diversity /= comparisons

            # Nouveauté (items peu populaires)
            item_popularity = pd.Series(test_data['item_ids']).value_counts()
            recommended_popularities = [item_popularity.get(item, 0) for item in set(all_recommendations)]
            novelty = 1 - (np.mean(recommended_popularities) / item_popularity.max()) if len(
                recommended_popularities) > 0 else 0.0

            return {
                'diversity': float(unique_items_ratio),
                'coverage': float(coverage),
                'inter_user_diversity': float(inter_user_diversity),
                'novelty': float(max(0, novelty))  # Assurer que la nouveauté est positive
            }

        except Exception as e:
            self.logger.warning(f"Erreur dans _evaluate_diversity: {e}")
            return {'diversity': 0.0, 'coverage': 0.0, 'novelty': 0.0}

    def compare_models(self, evaluation_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare les résultats de plusieurs modèles."""
        if not evaluation_results:
            return pd.DataFrame()

        try:
            # Créer un DataFrame de comparaison
            comparison_df = pd.DataFrame(evaluation_results).T

            # Ajouter un score composite
            if 'auc' in comparison_df.columns and 'precision_at_5' in comparison_df.columns:
                comparison_df['composite_score'] = (
                        0.4 * comparison_df['auc'] +
                        0.3 * comparison_df['precision_at_5'] +
                        0.2 * comparison_df['diversity'] +
                        0.1 * comparison_df['coverage']
                )

            # Trier par score composite
            if 'composite_score' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('composite_score', ascending=False)

            return comparison_df

        except Exception as e:
            self.logger.error(f"Erreur dans compare_models: {e}")
            return pd.DataFrame()

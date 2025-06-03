from models.model_config import AlgorithmConfig
import numpy as np
import pandas as pd

from models.models import RecommendationTrainer


def main():
    """Exemple d'utilisation du système de recommandation."""

    # Configuration
    config = AlgorithmConfig()

    # Simulation de données d'entraînement
    np.random.seed(config.random_state)

    n_users = 1000
    n_items = 500
    n_interactions = 10000

    # Génération de données simulées
    train_data = {
        'user_ids': np.random.randint(0, n_users, n_interactions),
        'item_ids': np.random.randint(0, n_items, n_interactions),
        'ratings': np.random.uniform(1, 5, n_interactions),
        'features': pd.DataFrame({
            'user_feature_1': np.random.randn(n_interactions),
            'user_feature_2': np.random.randn(n_interactions),
            'item_feature_1': np.random.randn(n_interactions),
            'sport': np.random.choice(['football', 'basketball', 'tennis'], n_interactions)
        }),
        'targets': np.random.randint(0, 2, n_interactions),
        'item_features': pd.DataFrame({
            'feature_1': np.random.randn(n_items),
            'feature_2': np.random.randn(n_items),
            'feature_3': np.random.randn(n_items)
        }),
        'interactions': pd.DataFrame({
            'user_id': np.random.randint(0, n_users, n_interactions),
            'item_id': np.random.randint(0, n_items, n_interactions),
            'rating': np.random.uniform(1, 5, n_interactions)
        })
    }

    val_data = {
        'user_ids': np.random.randint(0, n_users, 1000),
        'item_ids': np.random.randint(0, n_items, 1000),
        'ratings': np.random.uniform(1, 5, 1000)
    }

    # Entraînement
    trainer = RecommendationTrainer(config, None)
    trainer.train_all_algorithms(train_data, val_data)

    # Test des recommandations
    best_model = trainer.get_best_model('auc')
    if best_model:
        recommendations = best_model.recommend(user_id=123, n_recommendations=5)
        print(f"Recommandations pour l'utilisateur 123: {recommendations}")


if __name__ == "__main__":
    main()
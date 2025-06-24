import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
import mlflow.catboost
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import joblib
import time
from concurrent.futures import ThreadPoolExecutor
from helper.logging import logger
from models.model_config import AlgorithmConfig


class BaseRecommender(ABC):
    """Interface abstraite pour tous les algorithmes de recommandation."""

    def __init__(self, config: AlgorithmConfig, name: str):
        self.config = config
        self.name = name
        self.model = None
        self.is_trained = False
        self.metrics = {}
        self.training_time = 0

    @abstractmethod
    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None) -> 'BaseRecommender':
        """Entraîne le modèle."""
        pass

    @abstractmethod
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores pour les paires user-item."""
        pass

    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items pour un utilisateur."""
        pass

    def save(self, path: str):
        """Sauvegarde le modèle."""
        # Créer le répertoire si nécessaire
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'config': self.config,
            'name': self.name,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'training_time': self.training_time
        }
        joblib.dump(model_data, path)
        logger.info(f"{self.name} sauvegardé dans {path}")

    @classmethod
    def load(cls, path: str) -> 'BaseRecommender':
        """Charge un modèle sauvegardé."""
        model_data = joblib.load(path)
        instance = cls(model_data['config'], model_data['name'])
        instance.model = model_data['model']
        instance.is_trained = model_data['is_trained']
        instance.metrics = model_data.get('metrics', {})
        instance.training_time = model_data.get('training_time', 0)
        return instance


class CollaborativeFilteringRecommender(BaseRecommender):
    """Algorithme de filtrage collaboratif avec factorisation matricielle."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "CollaborativeFiltering")
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, train_data: Dict[str, Any],
            val_data: Optional[Dict[str, Any]] = None) -> 'CollaborativeFilteringRecommender':
        start_time = time.time()

        # Extraction des données
        user_ids = train_data['user_ids']
        item_ids = train_data['item_ids']
        ratings = []
        for r in train_data['ratings']:
            try:
                # Ensure r is a float to avoid string-float comparison issues
                ratings.append(float(r))
            except (ValueError, TypeError):
                ratings.append(0.0)
        ratings = np.array(ratings)

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

        # Création de la matrice sparse
        rating_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        # Entraînement avec PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')

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
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.cf_epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions = self.model(user_tensor, item_tensor)
            loss = self.model.compute_loss(predictions, rating_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Validation
            if val_data is not None and epoch % 10 == 0:
                val_loss = self._validate(val_data, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 10:  # Early stopping
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Extraction des facteurs entraînés
        self.user_factors = self.model.user_factors.weight.detach().cpu().numpy()
        self.item_factors = self.model.item_factors.weight.detach().cpu().numpy()
        self.user_biases = self.model.user_biases.weight.detach().cpu().numpy()
        self.item_biases = self.model.item_biases.weight.detach().cpu().numpy()
        self.global_bias = self.model.global_bias.item()

        self.is_trained = True
        self.training_time = time.time() - start_time

        logger.info(f"Collaborative Filtering entraîné en {self.training_time:.2f}s")
        return self

    def _validate(self, val_data: Dict[str, Any], device: torch.device) -> float:
        """Validation du modèle."""
        self.model.eval()

        val_user_ids = val_data['user_ids']
        val_item_ids = val_data['item_ids']
        val_ratings = val_data['ratings']

        # Conversion en indices
        val_user_indices = []
        val_item_indices = []
        val_ratings_filtered = []

        for i, (uid, iid, rating) in enumerate(zip(val_user_ids, val_item_ids, val_ratings)):
            if uid in self.user_mapping and iid in self.item_mapping:
                val_user_indices.append(self.user_mapping[uid])
                val_item_indices.append(self.item_mapping[iid])
                try:
                    val_ratings_filtered.append(float(rating))
                except (ValueError, TypeError):
                    val_ratings_filtered.append(0.0)

        if not val_user_indices:
            return float('inf')

        val_user_tensor = torch.LongTensor(val_user_indices).to(device)
        val_item_tensor = torch.LongTensor(val_item_indices).to(device)
        val_rating_tensor = torch.FloatTensor(val_ratings_filtered).to(device)

        with torch.no_grad():
            predictions = self.model(val_user_tensor, val_item_tensor)
            val_loss = F.mse_loss(predictions, val_rating_tensor)

        self.model.train()
        return val_loss.item()

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
                predictions.append(self.global_bias)  # Fallback

        return np.array(predictions)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items pour un utilisateur."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")

        if user_id not in self.user_mapping:
            return []  # Utilisateur inconnu

        user_idx = self.user_mapping[user_id]
        item_ids = list(self.item_mapping.keys())

        # Calcul des scores pour tous les items
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

        # Tri et sélection du top-N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]


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
        """Forward pass."""
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        # Calcul de la prédiction
        dot_product = (user_embedding * item_embedding).sum(dim=1)
        prediction = self.global_bias + user_bias + item_bias + dot_product

        return prediction

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcul de la loss avec régularisation."""
        mse_loss = F.mse_loss(predictions, targets)

        # Régularisation L2
        reg_loss = (
                self.user_factors.weight.norm(2) +
                self.item_factors.weight.norm(2) +
                self.user_biases.weight.norm(2) +
                self.item_biases.weight.norm(2)
        )

        return mse_loss + self.reg_lambda * reg_loss


class ContentBasedRecommender(BaseRecommender):
    """Algorithme de filtrage basé sur le contenu avec réseau de neurones."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "ContentBased")
        self.item_features = None
        self.user_profile_model = None
        self.item_encoder = None
        self.scaler = StandardScaler()

    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None) -> 'ContentBasedRecommender':
        start_time = time.time()

        # Extraction des features
        self.item_features = train_data['item_features']  # DataFrame des features d'items ou matrice sparse
        user_item_interactions = train_data['interactions']  # DataFrame user_id, item_id, rating

        # Normalisation des features
        if isinstance(self.item_features, pd.DataFrame):
            # Si c'est un DataFrame, utiliser select_dtypes
            item_feature_matrix = self.scaler.fit_transform(self.item_features.select_dtypes(include=[np.number]))
        elif isinstance(self.item_features, csr_matrix):
            # Si c'est une matrice sparse, la convertir en array dense pour la normalisation
            item_feature_matrix = self.scaler.fit_transform(self.item_features.toarray())
        else:
            # Autre cas (numpy array, etc.)
            item_feature_matrix = self.scaler.fit_transform(self.item_features)

        # Construction des profils utilisateurs
        user_profiles = self._build_user_profiles(user_item_interactions, item_feature_matrix)

        # Préparation des données d'entraînement
        X_train, y_train = self._prepare_training_data(user_item_interactions, user_profiles, item_feature_matrix)

        # Création du modèle
        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')

        input_dim = X_train.shape[1]
        self.model = ContentBasedModel(
            input_dim=input_dim,
            hidden_dims=self.config.cb_hidden_dims,
            dropout=self.config.cb_dropout
        ).to(device)

        # Entraînement
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.cb_batch_size,
            shuffle=True,
            num_workers=4
        )

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.cb_epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        self.is_trained = True
        self.training_time = time.time() - start_time

        # Stockage des profils utilisateurs
        self.user_profiles = user_profiles

        logger.info(f"Content-Based Filtering entraîné en {self.training_time:.2f}s")
        return self

    def _build_user_profiles(self, interactions: pd.DataFrame, item_features: np.ndarray) -> Dict[int, np.ndarray]:
        """Construit les profils utilisateurs basés sur les interactions."""
        user_profiles = {}

        for user_id in interactions['user_id'].unique():
            user_interactions = interactions[interactions['user_id'] == user_id]

            # Moyenne pondérée des features des items avec lesquels l'utilisateur a interagi
            weighted_features = np.zeros(item_features.shape[1])
            total_weight = 0

            for _, row in user_interactions.iterrows():
                item_idx = row['item_id']  # Suppose que item_id correspond à l'index
                rating = row['rating']

                try:
                    item_idx = int(item_idx)
                    rating = float(rating)
                    if item_idx < len(item_features):
                        # Handle different types of item_features
                        if isinstance(item_features, pd.DataFrame):
                            item_feature = item_features.iloc[item_idx].values
                        elif isinstance(item_features, csr_matrix):
                            item_feature = item_features[item_idx].toarray()[0]
                        elif isinstance(item_features, np.ndarray):
                            item_feature = item_features[item_idx]
                        else:
                            # Skip unknown types
                            continue

                        weighted_features += rating * item_feature
                        total_weight += rating
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error in _build_user_profiles for item {item_idx}: {e}")
                    continue

            if total_weight > 0:
                user_profiles[user_id] = weighted_features / total_weight
            else:
                user_profiles[user_id] = np.zeros(item_features.shape[1])

        return user_profiles

    def _prepare_training_data(self, interactions: pd.DataFrame, user_profiles: Dict, item_features: np.ndarray) -> \
    Tuple[np.ndarray, np.ndarray]:
        """Prépare les données d'entraînement."""
        X = []
        y = []

        for _, row in interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']

            try:
                item_id = int(item_id)
                rating = float(rating)
                if user_id in user_profiles and item_id < len(item_features):
                    # Concaténation du profil utilisateur et des features de l'item
                    user_profile = user_profiles[user_id]

                    # Handle different types of item_features
                    if isinstance(item_features, pd.DataFrame):
                        item_feature = item_features.iloc[item_id].values
                    elif isinstance(item_features, csr_matrix):
                        item_feature = item_features[item_id].toarray()[0]
                    elif isinstance(item_features, np.ndarray):
                        item_feature = item_features[item_id]
                    else:
                        # Skip unknown types
                        continue

                    combined_features = np.concatenate([user_profile, item_feature])
                    X.append(combined_features)
                    y.append(rating)
            except (ValueError, TypeError, IndexError) as e:
                logger.debug(f"Error in _prepare_training_data for item {item_id}: {e}")
                continue

        return np.array(X), np.array(y)

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores pour les paires user-item."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for user_id, item_id in zip(user_ids, item_ids):
                try:
                    item_id = int(item_id)
                    if user_id in self.user_profiles and item_id < len(self.item_features):
                        user_profile = self.user_profiles[user_id]

                        # Handle different types of item_features
                        if isinstance(self.item_features, pd.DataFrame):
                            item_feature = self.scaler.transform([self.item_features.iloc[item_id].values])[0]
                        elif isinstance(self.item_features, csr_matrix):
                            item_feature = self.scaler.transform([self.item_features[item_id].toarray()[0]])[0]
                        elif isinstance(self.item_features, np.ndarray):
                            item_feature = self.scaler.transform([self.item_features[item_id]])[0]
                        else:
                            # Fallback for unknown type
                            predictions.append(0.0)
                            continue

                        combined_features = np.concatenate([user_profile, item_feature])
                        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)

                        prediction = self.model(features_tensor).item()
                        predictions.append(prediction)
                    else:
                        predictions.append(0.0)  # Fallback
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error in predict: {e}")
                    predictions.append(0.0)  # Fallback for errors

        return np.array(predictions)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items pour un utilisateur."""
        if not self.is_trained or user_id not in self.user_profiles:
            return []

        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')
        self.model.eval()

        user_profile = self.user_profiles[user_id]
        scores = []

        with torch.no_grad():
            for item_id in range(len(self.item_features)):
                try:
                    # Handle different types of item_features
                    if isinstance(self.item_features, pd.DataFrame):
                        item_feature = self.scaler.transform([self.item_features.iloc[item_id].values])[0]
                    elif isinstance(self.item_features, csr_matrix):
                        item_feature = self.scaler.transform([self.item_features[item_id].toarray()[0]])[0]
                    elif isinstance(self.item_features, np.ndarray):
                        item_feature = self.scaler.transform([self.item_features[item_id]])[0]
                    else:
                        # Skip unknown types
                        continue

                    combined_features = np.concatenate([user_profile, item_feature])
                    features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)

                    score = self.model(features_tensor).item()
                    scores.append((item_id, score))
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error in recommend for item {item_id}: {e}")
                    continue

        # Tri et sélection du top-N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]


class ContentBasedModel(nn.Module):
    """Modèle de réseau de neurones pour le filtrage basé sur le contenu."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HybridRecommender(BaseRecommender):
    """Système de recommandation hybride combinant plusieurs approches."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "Hybrid")
        self.recommenders = {}
        self.weights = config.hybrid_weights
        self.ensemble_method = config.hybrid_ensemble_method

    def fit(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None) -> 'HybridRecommender':
        start_time = time.time()

        # Entraînement en parallèle des sous-modèles
        if self.config.enable_parallel_training:
            self._fit_parallel(train_data, val_data)
        else:
            self._fit_sequential(train_data, val_data)

        self.is_trained = True
        self.training_time = time.time() - start_time

        logger.info(f"Hybrid Recommender entraîné en {self.training_time:.2f}s")
        return self

    def _fit_parallel(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]]):
        """Entraînement en parallèle des sous-modèles."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            # Collaborative Filtering
            if 'collaborative' in self.weights:
                cf_recommender = CollaborativeFilteringRecommender(self.config)
                futures['collaborative'] = executor.submit(cf_recommender.fit, train_data, val_data)

            # Content-Based Filtering
            if 'content_based' in self.weights:
                cb_recommender = ContentBasedRecommender(self.config)
                futures['content_based'] = executor.submit(cb_recommender.fit, train_data, val_data)

            # Contextual (implémentation simplifiée avec CatBoost)
            if 'contextual' in self.weights:
                contextual_recommender = ContextualCatBoostRecommender(self.config)
                futures['contextual'] = executor.submit(contextual_recommender.fit, train_data, val_data)

            # Récupération des résultats
            for name, future in futures.items():
                try:
                    self.recommenders[name] = future.result()
                    logger.info(f"Sous-modèle {name} entraîné avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'entraînement du sous-modèle {name}: {e}")

    def _fit_sequential(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]]):
        """Entraînement séquentiel des sous-modèles."""
        if 'collaborative' in self.weights:
            self.recommenders['collaborative'] = CollaborativeFilteringRecommender(self.config).fit(train_data,
                                                                                                    val_data)

        if 'content_based' in self.weights:
            self.recommenders['content_based'] = ContentBasedRecommender(self.config).fit(train_data, val_data)

        if 'contextual' in self.weights:
            self.recommenders['contextual'] = ContextualCatBoostRecommender(self.config).fit(train_data, val_data)

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores en combinant les prédictions des sous-modèles."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        predictions = {}

        # Collecte des prédictions de chaque sous-modèle
        for name, recommender in self.recommenders.items():
            try:
                predictions[name] = recommender.predict(user_ids, item_ids, context)
            except Exception as e:
                logger.warning(f"Erreur lors de la prédiction avec {name}: {e}")
                predictions[name] = np.zeros(len(user_ids))

        # Combinaison des prédictions
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_ensemble(predictions)
        elif self.ensemble_method == 'stacking':
            return self._stacking_ensemble(predictions, user_ids, item_ids)
        else:
            return self._weighted_average_ensemble(predictions)

    def _weighted_average_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine les prédictions par moyenne pondérée."""
        combined = np.zeros(len(list(predictions.values())[0]))
        total_weight = 0

        for name, preds in predictions.items():
            weight = self.weights.get(name, 0)
            combined += weight * preds
            total_weight += weight

        return combined / max(total_weight, 1e-8)

    def _stacking_ensemble(self, predictions: Dict[str, np.ndarray], user_ids: np.ndarray,
                           item_ids: np.ndarray) -> np.ndarray:
        """Combine les prédictions avec un meta-learner."""

        return self._weighted_average_ensemble(predictions)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande les top-N items en combinant les recommandations."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")

        # Collecte des recommandations de chaque sous-modèle
        all_recommendations = {}

        for name, recommender in self.recommenders.items():
            try:
                recs = recommender.recommend(user_id, n_recommendations * 2, context)
                all_recommendations[name] = recs
            except Exception as e:
                logger.warning(f"Erreur lors des recommandations avec {name}: {e}")
                all_recommendations[name] = []

        # Fusion des recommandations
        item_scores = {}

        for name, recs in all_recommendations.items():
            weight = self.weights.get(name, 0)
            for item_id, score in recs:
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                item_scores[item_id] += weight * score

        # Tri et sélection du top-N
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


class ContextualCatBoostRecommender(BaseRecommender):
    """Recommandeur contextuel utilisant CatBoost."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "ContextualCatBoost")
        self.feature_columns = None
        self.categorical_features = None

    def fit(self, train_data: Dict[str, Any],
            val_data: Optional[Dict[str, Any]] = None) -> 'ContextualCatBoostRecommender':
        start_time = time.time()

        # Extraction des données
        X_train = train_data['features']  # DataFrame avec features contextuelles
        y_train = train_data['targets']  # Array des targets

        # Vérifier que X_train est bien un DataFrame
        if not isinstance(X_train, pd.DataFrame):
            # Convertir en DataFrame si ce n'est pas déjà le cas
            if hasattr(X_train, 'toarray'):  # Pour les matrices sparse
                X_train = pd.DataFrame(X_train.toarray())
            else:
                X_train = pd.DataFrame(X_train)
            # Mettre à jour les données d'entraînement
            train_data['features'] = X_train

        # Fonction pour gérer les noms de colonnes dupliqués
        def handle_duplicate_columns(df):
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                logger.warning(f"Colonnes dupliquées détectées: {duplicate_cols}")
                # Renommer les colonnes dupliquées
                for col in duplicate_cols:
                    dup_indices = [i for i, c in enumerate(df.columns) if c == col]
                    for i, idx in enumerate(dup_indices[1:], 1):  # Skip the first occurrence
                        new_col_name = f"{col}_{i}"
                        df.columns.values[idx] = new_col_name
                        logger.info(f"Colonne renommée: {col} -> {new_col_name}")
            return df

        # Gérer les noms de colonnes dupliqués dans les données d'entraînement
        X_train = handle_duplicate_columns(X_train)
        train_data['features'] = X_train

        # Gérer les noms de colonnes dupliqués dans les données de validation si présentes
        if val_data is not None and 'features' in val_data:
            X_val = val_data['features']
            if isinstance(X_val, pd.DataFrame):
                X_val = handle_duplicate_columns(X_val)
                # S'assurer que les colonnes de validation correspondent à celles d'entraînement
                if set(X_val.columns) != set(X_train.columns):
                    logger.warning("Les colonnes de validation ne correspondent pas aux colonnes d'entraînement")
                    # Ajouter les colonnes manquantes avec des valeurs par défaut
                    for col in X_train.columns:
                        if col not in X_val.columns:
                            X_val[col] = 0
                    # Réordonner les colonnes pour correspondre à X_train
                    X_val = X_val[X_train.columns]
                val_data['features'] = X_val

        # Identification des features catégorielles
        self.categorical_features = []
        for col in X_train.columns:
            try:
                if pd.api.types.is_object_dtype(X_train[col]):
                    self.categorical_features.append(col)
            except:
                # En cas d'erreur, ignorer cette colonne
                pass

        self.feature_columns = X_train.columns.tolist()

        # Préparation des données de validation
        eval_set = None
        if val_data is not None:
            X_val = val_data['features']
            y_val = val_data['targets']
            eval_set = (X_val, y_val)

        # Configuration du modèle CatBoost
        self.model = CatBoostClassifier(
            iterations=self.config.catboost_iterations,
            learning_rate=self.config.catboost_learning_rate,
            depth=self.config.catboost_depth,
            l2_leaf_reg=self.config.catboost_l2_leaf_reg,
            early_stopping_rounds=self.config.catboost_early_stopping_rounds,
            random_seed=self.config.random_state,
            verbose=100,
            task_type='GPU' if self.config.enable_gpu and torch.cuda.is_available() else 'CPU'
        )

        # Entraînement
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            cat_features=self.categorical_features,
            use_best_model=True if eval_set else False
        )

        self.is_trained = True
        self.training_time = time.time() - start_time

        # Calcul des feature importances
        self.feature_importance = dict(zip(self.feature_columns, self.model.get_feature_importance()))

        logger.info(f"Contextual CatBoost entraîné en {self.training_time:.2f}s")
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores avec les features contextuelles."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        # Construction du DataFrame de features pour la prédiction
        # Cette partie dépend de la structure des données contextuelles
        # Implémentation simplifiée
        if context and 'features' in context:
            X_pred = context['features']
            return self.model.predict_proba(X_pred)[:, 1]
        else:
            # Fallback avec scores neutres
            return np.full(len(user_ids), 0.5)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande en utilisant le contexte."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")

        # Implémentation dépendante du contexte spécifique
        # Retourne une liste vide par défaut
        return []


class ContextualBanditRecommender(BaseRecommender):
    """Bandit contextuel pour l'exploration-exploitation."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "ContextualBandit")
        self.action_counts = {}
        self.action_rewards = {}
        self.context_dim = config.bandit_context_dim
        self.alpha = config.bandit_alpha
        self.exploration_rate = config.bandit_exploration_rate
        self.update_frequency = config.bandit_update_frequency
        self.updates_count = 0

    def fit(self, train_data: Dict[str, Any],
            val_data: Optional[Dict[str, Any]] = None) -> 'ContextualBanditRecommender':
        start_time = time.time()

        # Initialisation des actions possibles
        actions = train_data.get('actions', [])
        for action in actions:
            self.action_counts[action] = 0
            self.action_rewards[action] = 0.0

        # Le bandit apprend en ligne, pas d'entraînement batch traditionnel
        self.is_trained = True
        self.training_time = time.time() - start_time

        logger.info(f"Contextual Bandit initialisé avec {len(actions)} actions")
        return self

    def select_action(self, context: Dict[str, Any], available_actions: List[int]) -> int:
        """Sélectionne une action selon la stratégie UCB ou epsilon-greedy."""
        if np.random.random() < self.exploration_rate:
            # Exploration
            return np.random.choice(available_actions)

        # Exploitation
        best_action = None
        best_score = float('-inf')

        total_counts = sum(self.action_counts.values())

        for action in available_actions:
            if self.action_counts.get(action, 0) == 0:
                return action  # Sélection des actions non testées

            avg_reward = self.action_rewards[action] / self.action_counts[action]

            # UCB score
            if total_counts > 0:
                exploration_bonus = self.alpha * np.sqrt(2 * np.log(total_counts) / self.action_counts[action])
                score = avg_reward + exploration_bonus
            else:
                score = avg_reward

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else np.random.choice(available_actions)

    def update_reward(self, action: int, reward: float):
        """Met à jour les récompenses pour une action."""
        if action not in self.action_counts:
            self.action_counts[action] = 0
            self.action_rewards[action] = 0.0

        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.updates_count += 1

        # Décroissance de l'exploration
        if self.updates_count % self.update_frequency == 0:
            self.exploration_rate *= 0.99

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les scores basés sur les statistiques du bandit."""
        predictions = []

        for user_id, item_id in zip(user_ids, item_ids):
            if item_id in self.action_rewards and self.action_counts[item_id] > 0:
                avg_reward = self.action_rewards[item_id] / self.action_counts[item_id]
                predictions.append(avg_reward)
            else:
                predictions.append(0.5)  # Score neutre pour les actions inconnues

        return np.array(predictions)

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande en utilisant la stratégie du bandit."""
        available_actions = list(self.action_counts.keys())

        if not available_actions:
            return []

        # Sélection des top actions
        action_scores = []
        for action in available_actions:
            if self.action_counts[action] > 0:
                score = self.action_rewards[action] / self.action_counts[action]
            else:
                score = 0.5
            action_scores.append((action, score))

        # Tri et sélection
        action_scores.sort(key=lambda x: x[1], reverse=True)
        return action_scores[:n_recommendations]


class ReinforcementLearningRecommender(BaseRecommender):
    """Recommandeur basé sur l'apprentissage par renforcement (DQN)."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config, "ReinforcementLearning")
        self.state_dim = config.rl_state_dim
        self.action_dim = config.rl_action_dim
        self.memory = []
        self.memory_size = config.rl_memory_size
        self.epsilon = config.rl_epsilon
        self.epsilon_decay = config.rl_epsilon_decay
        self.gamma = config.rl_gamma

    def fit(self, train_data: Dict[str, Any],
            val_data: Optional[Dict[str, Any]] = None) -> 'ReinforcementLearningRecommender':
        start_time = time.time()

        device = torch.device('cuda' if torch.cuda.is_available() and self.config.enable_gpu else 'cpu')

        # Création du réseau Q
        self.model = DQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.rl_hidden_dims
        ).to(device)

        self.target_model = DQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.config.rl_hidden_dims
        ).to(device)

        # Copie des poids vers le modèle cible
        self.target_model.load_state_dict(self.model.state_dict())

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.rl_learning_rate)

        # Entraînement par épisodes (simulation)
        episodes = train_data.get('episodes', [])

        for episode in episodes:
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            next_states = episode['next_states']
            dones = episode['dones']

            # Stockage des expériences
            for i in range(len(states)):
                self.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])

            # Entraînement si assez d'expériences
            if len(self.memory) > self.config.rl_batch_size:
                self._replay(optimizer, device)

            # Décroissance epsilon
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        self.is_trained = True
        self.training_time = time.time() - start_time

        logger.info(f"RL Recommender entraîné en {self.training_time:.2f}s")
        return self

    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire."""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def _replay(self, optimizer, device):
        """Entraîne le modèle sur un batch d'expériences."""
        batch_size = min(self.config.rl_batch_size, len(self.memory))
        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, context: Optional[Dict] = None) -> np.ndarray:
        """Prédit les Q-values pour les paires user-item."""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        # Implémentation simplifiée - retourne des scores aléatoires
        return np.random.random(len(user_ids))

    def recommend(self, user_id: int, n_recommendations: int = 10, context: Optional[Dict] = None) -> List[
        Tuple[int, float]]:
        """Recommande en utilisant la politique epsilon-greedy."""
        if not self.is_trained:
            return []

        # Implémentation simplifiée
        actions = list(range(100))  # Actions possibles
        scores = np.random.random(len(actions))

        action_scores = list(zip(actions, scores))
        action_scores.sort(key=lambda x: x[1], reverse=True)

        return action_scores[:n_recommendations]


class DQNModel(nn.Module):
    """Réseau de neurones pour Deep Q-Learning."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class RecommendationTrainer:
    """Gestionnaire d'entraînement pour tous les algorithmes."""

    def __init__(self, config: AlgorithmConfig, data_processor):
        self.config = config
        self.data_processor = data_processor
        self.trained_models = {}

        # Configuration MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)

    def train_all_algorithms(self, train_data: Dict[str, Any], val_data: Optional[Dict[str, Any]] = None):
        """Entraîne tous les algorithmes."""
        algorithms = {
            'collaborative_filtering': CollaborativeFilteringRecommender,
            'content_based': ContentBasedRecommender,
            'hybrid': HybridRecommender,
            'contextual_bandit': ContextualBanditRecommender,
            'reinforcement_learning': ReinforcementLearningRecommender
        }

        for name, algorithm_class in algorithms.items():
            logger.info(f"Entraînement de {name}...")

            with mlflow.start_run(run_name=f"{name}_training"):
                try:
                    recommender = algorithm_class(self.config)
                    recommender.fit(train_data, val_data)

                    # Évaluation
                    metrics = self._evaluate_model(recommender, val_data)

                    # Logging MLflow
                    mlflow.log_params(recommender.config.__dict__)
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric("training_time", recommender.training_time)

                    self.trained_models[name] = recommender

                    # Sauvegarde
                    model_path = f"models/{name}_model.joblib"
                    recommender.save(model_path)
                    mlflow.log_artifact(model_path)

                    logger.info(f"{name} entraîné avec succès. Métriques: {metrics}")

                except Exception as e:
                    logger.error(f"Erreur lors de l'entraînement de {name}: {e}")
                    mlflow.log_param("error", str(e))

    def _evaluate_model(self, model: BaseRecommender, val_data: Dict[str, Any]) -> Dict[str, float]:
        """Évalue un modèle sur les données de validation."""
        if val_data is None:
            return {}

        try:
            # Évaluation basique
            user_ids = val_data.get('user_ids', [])
            item_ids = val_data.get('item_ids', [])
            true_ratings = val_data.get('ratings', [])

            if len(user_ids) == 0:
                return {}

            predicted_ratings = model.predict(np.array(user_ids), np.array(item_ids))

            # Calcul des métriques
            mse = np.mean((true_ratings - predicted_ratings) ** 2)
            mae = np.mean(np.abs(true_ratings - predicted_ratings))

            # Conversion en classification binaire pour AUC
            binary_true = (np.array(true_ratings) > np.mean(true_ratings)).astype(int)

            try:
                auc = roc_auc_score(binary_true, predicted_ratings)
            except:
                auc = 0.5

            return {
                'mse': float(mse),
                'mae': float(mae),
                'auc': float(auc)
            }

        except Exception as e:
            logger.warning(f"Erreur lors de l'évaluation: {e}")
            return {}

    def get_best_model(self, metric: str = 'auc') -> BaseRecommender:
        """Retourne le meilleur modèle selon une métrique."""
        if not self.trained_models:
            raise ValueError("Aucun modèle entraîné")

        best_model = None
        best_score = float('-inf') if metric in ['auc'] else float('inf')

        for name, model in self.trained_models.items():
            if metric in model.metrics:
                score = model.metrics[metric]

                if metric in ['auc'] and score > best_score:
                    best_score = score
                    best_model = model
                elif metric in ['mse', 'mae'] and score < best_score:
                    best_score = score
                    best_model = model

        return best_model

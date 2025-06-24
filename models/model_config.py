from dataclasses import dataclass, field
from typing import List, Dict


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
    mlflow_tracking_uri: str = "http://51.91.128.26:5000"
    mlflow_experiment_name: str = "betting_recommendation"

    # Performance
    enable_gpu: bool = True
    enable_mixed_precision: bool = True
    enable_parallel_training: bool = False

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


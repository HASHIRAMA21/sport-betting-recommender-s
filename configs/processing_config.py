from dataclasses import dataclass, field
from typing import List


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
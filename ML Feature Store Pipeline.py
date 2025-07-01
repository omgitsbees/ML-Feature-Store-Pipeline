import pandas as pd
import sqlite3
import hashlib
import yaml
import asyncio
import aiosqlite
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps
import time
from pydantic import BaseModel, validator
import numpy as np

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Models ---
class FeatureConfig(BaseModel):
    name: str
    dtype: str
    description: str
    tags: List[str] = []
    owner: str = "unknown"
    
    @validator('dtype')
    def validate_dtype(cls, v):
        valid_types = ['int64', 'float64', 'object', 'datetime64[ns]', 'bool']
        if v not in valid_types:
            raise ValueError(f'dtype must be one of {valid_types}')
        return v

class FeatureMetadata(BaseModel):
    feature_version: str
    description: str
    created_at: str
    features: List[FeatureConfig]
    data_quality_score: float = 1.0
    lineage: Dict[str, Any] = {}
    tags: List[str] = []

@dataclass
class DataQualityMetrics:
    null_percentage: float
    duplicate_percentage: float
    outlier_percentage: float
    schema_violations: int
    overall_score: float

# --- Interfaces ---
class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_feature_configs(self) -> List[FeatureConfig]:
        pass

class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

# --- Cache Implementations ---
class InMemoryCache(CacheBackend):
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                timestamp, ttl = self._timestamps[key]
                if time.time() - timestamp < ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = (time.time(), ttl)
    
    async def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)

# --- Data Quality ---
class DataQualityValidator:
    def __init__(self):
        self.checks = []
    
    def add_check(self, check_func: Callable[[pd.DataFrame], bool], name: str):
        self.checks.append((check_func, name))
    
    def validate(self, df: pd.DataFrame) -> DataQualityMetrics:
        null_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_pct = df.duplicated().sum() / len(df)
        
        # Simple outlier detection using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = 0
        total_numeric = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            outliers += outlier_mask.sum()
            total_numeric += len(df[col].dropna())
        
        outlier_pct = outliers / max(total_numeric, 1)
        
        # Schema validation
        violations = 0
        for check_func, name in self.checks:
            try:
                if not check_func(df):
                    violations += 1
                    logger.warning(f"Data quality check failed: {name}")
            except Exception as e:
                violations += 1
                logger.error(f"Data quality check error for {name}: {e}")
        
        # Calculate overall score
        score = max(0, 1 - (null_pct + duplicate_pct + outlier_pct + violations * 0.1))
        
        return DataQualityMetrics(
            null_percentage=null_pct,
            duplicate_percentage=duplicate_pct,
            outlier_percentage=outlier_pct,
            schema_violations=violations,
            overall_score=score
        )

# --- Feature Extractors ---
class UserEventExtractor(FeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.groupby("user_id").agg(
            total_events=pd.NamedAgg(column="event_type", aggfunc="count"),
            total_purchases=pd.NamedAgg(column="amount", aggfunc=lambda x: (x > 0).sum()),
            total_amount=pd.NamedAgg(column="amount", aggfunc="sum"),
            avg_amount=pd.NamedAgg(column="amount", aggfunc=lambda x: x[x > 0].mean()),
            last_event_time=pd.NamedAgg(column="timestamp", aggfunc="max"),
            first_event_time=pd.NamedAgg(column="timestamp", aggfunc="min"),
            unique_event_types=pd.NamedAgg(column="event_type", aggfunc="nunique"),
            days_active=pd.NamedAgg(column="timestamp", aggfunc=lambda x: (x.max() - x.min()).days + 1)
        ).reset_index()
        
        # Fill NaN values
        features["avg_amount"] = features["avg_amount"].fillna(0)
        
        # Add derived features
        features["purchase_rate"] = features["total_purchases"] / features["total_events"]
        features["avg_events_per_day"] = features["total_events"] / features["days_active"]
        
        return features
    
    def get_feature_configs(self) -> List[FeatureConfig]:
        return [
            FeatureConfig(name="user_id", dtype="int64", description="Unique user identifier"),
            FeatureConfig(name="total_events", dtype="int64", description="Total number of events", tags=["count"]),
            FeatureConfig(name="total_purchases", dtype="int64", description="Total number of purchases", tags=["count", "revenue"]),
            FeatureConfig(name="total_amount", dtype="float64", description="Total purchase amount", tags=["revenue"]),
            FeatureConfig(name="avg_amount", dtype="float64", description="Average purchase amount", tags=["revenue"]),
            FeatureConfig(name="last_event_time", dtype="datetime64[ns]", description="Timestamp of last event", tags=["temporal"]),
            FeatureConfig(name="first_event_time", dtype="datetime64[ns]", description="Timestamp of first event", tags=["temporal"]),
            FeatureConfig(name="unique_event_types", dtype="int64", description="Number of unique event types", tags=["diversity"]),
            FeatureConfig(name="days_active", dtype="int64", description="Number of days user has been active", tags=["temporal"]),
            FeatureConfig(name="purchase_rate", dtype="float64", description="Ratio of purchases to total events", tags=["derived", "revenue"]),
            FeatureConfig(name="avg_events_per_day", dtype="float64", description="Average events per day", tags=["derived", "engagement"])
        ]

# --- Monitoring ---
class FeatureMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []
    
    def log_feature_access(self, feature_version: str, user_id: Optional[int] = None):
        key = f"access_{feature_version}"
        self.metrics[key] = self.metrics.get(key, 0) + 1
        logger.info(f"Feature access logged: {feature_version} for user {user_id}")
    
    def log_feature_creation(self, feature_version: str, quality_score: float):
        self.metrics[f"creation_{feature_version}"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "quality_score": quality_score
        }
        
        if quality_score < 0.8:
            alert = f"Low quality features detected: {feature_version} (score: {quality_score})"
            self.alerts.append(alert)
            logger.warning(alert)
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def get_alerts(self) -> List[str]:
        return self.alerts

# --- Advanced Feature Store ---
class AdvancedFeatureStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config["feature_store_db"]
        self.feature_table = config["feature_table"]
        self.metadata_table = config["feature_metadata_table"]
        self.cache = InMemoryCache()
        self.monitor = FeatureMonitor()
        self.validator = DataQualityValidator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Add default data quality checks
        self._setup_default_quality_checks()
        
        # Initialize database
        asyncio.run(self._init_database())
    
    def _setup_default_quality_checks(self):
        self.validator.add_check(
            lambda df: len(df) > 0,
            "non_empty_dataframe"
        )
        self.validator.add_check(
            lambda df: not df.isnull().all().any(),
            "no_completely_null_columns"
        )
        self.validator.add_check(
            lambda df: "user_id" in df.columns,
            "user_id_column_exists"
        )
    
    async def _init_database(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.feature_table} (
                    user_id INTEGER,
                    total_events INTEGER,
                    total_purchases INTEGER,
                    total_amount REAL,
                    avg_amount REAL,
                    last_event_time TEXT,
                    first_event_time TEXT,
                    unique_event_types INTEGER,
                    days_active INTEGER,
                    purchase_rate REAL,
                    avg_events_per_day REAL,
                    feature_version TEXT,
                    created_at TEXT,
                    INDEX(user_id),
                    INDEX(feature_version)
                )
            """)
            
            await db.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                    feature_version TEXT PRIMARY KEY,
                    description TEXT,
                    created_at TEXT,
                    features_config TEXT,
                    data_quality_metrics TEXT,
                    lineage TEXT,
                    tags TEXT
                )
            """)
            await db.commit()
    
    async def register_features_async(
        self, 
        features: pd.DataFrame, 
        metadata: FeatureMetadata
    ) -> str:
        """Asynchronously register features with validation and monitoring"""
        
        # Validate data quality
        quality_metrics = self.validator.validate(features)
        metadata.data_quality_score = quality_metrics.overall_score
        
        # Generate feature version hash
        feature_hash = hashlib.md5(
            pd.util.hash_pandas_object(features).values
        ).hexdigest()
        metadata.feature_version = feature_hash
        
        # Add metadata columns to features
        features = features.copy()
        features["feature_version"] = feature_hash
        features["created_at"] = metadata.created_at
        
        async with aiosqlite.connect(self.db_path) as db:
            # Insert features
            features_dict = features.to_dict('records')
            placeholders = ', '.join(['?' for _ in features.columns])
            columns = ', '.join(features.columns)
            
            await db.executemany(
                f"INSERT INTO {self.feature_table} ({columns}) VALUES ({placeholders})",
                [tuple(row.values()) for row in features_dict]
            )
            
            # Insert metadata
            await db.execute(f"""
                INSERT OR REPLACE INTO {self.metadata_table} 
                (feature_version, description, created_at, features_config, data_quality_metrics, lineage, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.feature_version,
                metadata.description,
                metadata.created_at,
                json.dumps([f.dict() for f in metadata.features]),
                json.dumps(asdict(quality_metrics)),
                json.dumps(metadata.lineage),
                json.dumps(metadata.tags)
            ))
            
            await db.commit()
        
        # Log monitoring metrics
        self.monitor.log_feature_creation(feature_hash, quality_metrics.overall_score)
        
        # Cache the features
        cache_key = f"features_{feature_hash}"
        await self.cache.set(cache_key, features, ttl=3600)
        
        logger.info(f"Features registered successfully: {feature_hash}")
        return feature_hash
    
    def register_features(
        self, 
        features: pd.DataFrame, 
        metadata: FeatureMetadata
    ) -> str:
        """Synchronous wrapper for feature registration"""
        return asyncio.run(self.register_features_async(features, metadata))
    
    async def get_features_async(
        self, 
        feature_version: Optional[str] = None,
        user_ids: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get features with caching support"""
        
        if feature_version is None:
            # Get latest version
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    f"SELECT feature_version FROM {self.metadata_table} ORDER BY created_at DESC LIMIT 1"
                ) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        return pd.DataFrame()
                    feature_version = row[0]
        
        cache_key = f"features_{feature_version}"
        if user_ids:
            cache_key += f"_users_{'_'.join(map(str, user_ids))}"
        
        # Try cache first
        if use_cache:
            cached_features = await self.cache.get(cache_key)
            if cached_features is not None:
                logger.info(f"Features retrieved from cache: {feature_version}")
                self.monitor.log_feature_access(feature_version)
                return cached_features
        
        # Query database
        query = f"SELECT * FROM {self.feature_table} WHERE feature_version = ?"
        params = [feature_version]
        
        if user_ids:
            placeholders = ', '.join(['?' for _ in user_ids])
            query += f" AND user_id IN ({placeholders})"
            params.extend(user_ids)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Cache the result
        if use_cache:
            await self.cache.set(cache_key, df, ttl=3600)
        
        self.monitor.log_feature_access(feature_version, user_ids[0] if user_ids else None)
        logger.info(f"Features retrieved from database: {feature_version}")
        return df
    
    def get_features(
        self, 
        feature_version: Optional[str] = None,
        user_ids: Optional[List[int]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Synchronous wrapper for feature retrieval"""
        return asyncio.run(self.get_features_async(feature_version, user_ids, use_cache))
    
    async def serve_features_async(
        self, 
        user_id: int, 
        feature_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serve features for real-time inference"""
        df = await self.get_features_async(feature_version, [user_id])
        if df.empty:
            return {}
        
        # Convert to dictionary, excluding metadata columns
        exclude_cols = ["feature_version", "created_at"]
        feature_dict = df.drop(columns=exclude_cols, errors='ignore').iloc[0].to_dict()
        
        self.monitor.log_feature_access(
            feature_version or "latest", 
            user_id
        )
        
        return feature_dict
    
    def serve_features(
        self, 
        user_id: int, 
        feature_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for feature serving"""
        return asyncio.run(self.serve_features_async(user_id, feature_version))
    
    async def get_feature_metadata_async(self, feature_version: str) -> Optional[FeatureMetadata]:
        """Get metadata for a specific feature version"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT * FROM {self.metadata_table} WHERE feature_version = ?",
                (feature_version,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                return FeatureMetadata(
                    feature_version=row[0],
                    description=row[1],
                    created_at=row[2],
                    features=json.loads(row[3]),
                    data_quality_score=json.loads(row[4])["overall_score"],
                    lineage=json.loads(row[5]),
                    tags=json.loads(row[6])
                )
    
    def get_feature_metadata(self, feature_version: str) -> Optional[FeatureMetadata]:
        """Synchronous wrapper for metadata retrieval"""
        return asyncio.run(self.get_feature_metadata_async(feature_version))
    
    async def list_feature_versions_async(self) -> List[Dict[str, Any]]:
        """List all feature versions with metadata"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT * FROM {self.metadata_table} ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "feature_version": row[0],
                        "description": row[1],
                        "created_at": row[2],
                        "data_quality_score": json.loads(row[4])["overall_score"],
                        "tags": json.loads(row[6])
                    }
                    for row in rows
                ]
    
    def list_feature_versions(self) -> List[Dict[str, Any]]:
        """Synchronous wrapper for version listing"""
        return asyncio.run(self.list_feature_versions_async())
    
    async def cleanup_old_versions_async(self, keep_n: int = 5):
        """Clean up old feature versions, keeping only the latest N"""
        versions = await self.list_feature_versions_async()
        if len(versions) <= keep_n:
            return
        
        versions_to_delete = versions[keep_n:]
        
        async with aiosqlite.connect(self.db_path) as db:
            for version in versions_to_delete:
                feature_version = version["feature_version"]
                await db.execute(
                    f"DELETE FROM {self.feature_table} WHERE feature_version = ?",
                    (feature_version,)
                )
                await db.execute(
                    f"DELETE FROM {self.metadata_table} WHERE feature_version = ?",
                    (feature_version,)
                )
                
                # Clear from cache
                await self.cache.delete(f"features_{feature_version}")
                
            await db.commit()
        
        logger.info(f"Cleaned up {len(versions_to_delete)} old feature versions")
    
    def cleanup_old_versions(self, keep_n: int = 5):
        """Synchronous wrapper for cleanup"""
        asyncio.run(self.cleanup_old_versions_async(keep_n))
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring metrics and alerts"""
        return {
            "metrics": self.monitor.get_metrics(),
            "alerts": self.monitor.get_alerts(),
            "cache_info": "In-memory cache active",
            "database_path": self.db_path
        }

# --- Configuration Management ---
def create_advanced_config():
    config = {
        "raw_data": "data/raw_events.csv",
        "feature_store_db": "advanced_feature_store.db",
        "feature_table": "features",
        "feature_metadata_table": "feature_metadata",
        "cache_ttl": 3600,
        "data_quality_threshold": 0.8,
        "cleanup_schedule": "daily",
        "monitoring": {
            "enable_alerts": True,
            "alert_thresholds": {
                "quality_score": 0.8,
                "null_percentage": 0.1
            }
        }
    }
    
    Path("config").mkdir(exist_ok=True)
    config_path = Path("config/advanced_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path

# --- Enhanced Data Generation ---
def generate_advanced_example_data():
    Path("data").mkdir(exist_ok=True)
    
    # Generate more realistic data
    np.random.seed(42)
    n_users = 100
    n_events = 1000
    
    user_ids = np.random.choice(range(1, n_users + 1), n_events)
    event_types = np.random.choice(
        ["click", "view", "purchase", "add_to_cart", "search"], 
        n_events, 
        p=[0.4, 0.3, 0.1, 0.15, 0.05]
    )
    
    # Generate amounts (only for purchases)
    amounts = np.where(
        event_types == "purchase",
        np.random.exponential(50, n_events),
        0
    )
    
    # Generate timestamps
    base_time = pd.Timestamp("2023-01-01")
    timestamps = [
        base_time + pd.Timedelta(hours=np.random.exponential(2))
        for _ in range(n_events)
    ]
    timestamps.sort()
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "event_type": event_types,
        "amount": amounts,
        "timestamp": timestamps
    })
    
    df.to_csv("data/raw_events.csv", index=False)
    logger.info(f"Generated {n_events} events for {n_users} users")

# --- Main Pipeline ---
async def main():
    # Generate example data
    generate_advanced_example_data()
    
    # Create config
    config_path = create_advanced_config()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize feature store
    store = AdvancedFeatureStore(config)
    
    # Load and process data
    df = pd.read_csv(config["raw_data"], parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} events")
    
    # Extract features using the extractor
    extractor = UserEventExtractor()
    features = extractor.extract(df)
    
    # Create metadata
    metadata = FeatureMetadata(
        feature_version="",  # Will be generated
        description="Advanced user-level event features with quality validation",
        created_at=datetime.utcnow().isoformat(),
        features=extractor.get_feature_configs(),
        lineage={
            "source_data": config["raw_data"],
            "extractor": "UserEventExtractor",
            "created_by": "advanced_pipeline"
        },
        tags=["user_features", "events", "behavioral"]
    )
    
    # Register features
    feature_version = await store.register_features_async(features, metadata)
    logger.info(f"Registered feature version: {feature_version}")
    
    # Demonstrate feature serving
    print("\n=== Feature Store Demo ===")
    
    # Get latest features
    latest_features = await store.get_features_async()
    print(f"\nLatest features shape: {latest_features.shape}")
    print(latest_features.head())
    
    # Serve features for specific user
    user_features = await store.serve_features_async(1)
    print(f"\nFeatures for user 1:")
    for key, value in user_features.items():
        print(f"  {key}: {value}")
    
    # List feature versions
    versions = await store.list_feature_versions_async()
    print(f"\nFeature versions:")
    for version in versions:
        print(f"  {version['feature_version'][:8]}... - {version['description']} (Quality: {version['data_quality_score']:.2f})")
    
    # Show monitoring dashboard
    dashboard = store.get_monitoring_dashboard()
    print(f"\nMonitoring Dashboard:")
    print(f"  Metrics: {dashboard['metrics']}")
    print(f"  Alerts: {dashboard['alerts']}")
    
    # Cleanup old versions (keeping latest 3)
    await store.cleanup_old_versions_async(keep_n=3)

if __name__ == "__main__":
    asyncio.run(main())
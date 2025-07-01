# Advanced Feature Store

A production-ready, async-first feature store implementation for machine learning pipelines with built-in data quality validation, caching, monitoring, and observability.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SQLite](https://img.shields.io/badge/database-SQLite-green.svg)](https://sqlite.org/)
[![Async](https://img.shields.io/badge/async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Core Capabilities
- **Async-first architecture** for high-performance feature operations
- **Data quality validation** with automated scoring and alerting
- **In-memory caching** with TTL for fast feature serving
- **Feature versioning** with lineage tracking and metadata management
- **Real-time monitoring** with metrics collection and alerting
- **Modular design** with pluggable extractors and cache backends

### Production Ready
- **Type safety** with Pydantic models and proper validation
- **Comprehensive logging** for debugging and observability
- **Error handling** with graceful recovery mechanisms
- **Resource management** with connection pooling and cleanup
- **Configuration management** with YAML-based settings

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Data Quality](#data-quality)
- [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites
- Python 3.8+
- SQLite 3.x

### Dependencies

```bash
pip install pandas numpy pydantic aiosqlite pyyaml
```

### Quick Install

```bash
git clone https://github.com/yourusername/advanced-feature-store.git
cd advanced-feature-store
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Setup

```python
import asyncio
from advanced_feature_store import AdvancedFeatureStore, create_advanced_config

# Create configuration
config_path = create_advanced_config()
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize feature store
store = AdvancedFeatureStore(config)
```

### 2. Extract and Register Features

```python
from advanced_feature_store import UserEventExtractor, FeatureMetadata
import pandas as pd

# Load your data
df = pd.read_csv("your_events.csv", parse_dates=["timestamp"])

# Extract features
extractor = UserEventExtractor()
features = extractor.extract(df)

# Create metadata
metadata = FeatureMetadata(
    feature_version="",  # Auto-generated
    description="User behavioral features",
    created_at=datetime.utcnow().isoformat(),
    features=extractor.get_feature_configs(),
    tags=["user_features", "behavioral"]
)

# Register features
feature_version = await store.register_features_async(features, metadata)
```

### 3. Serve Features for ML

```python
# Real-time feature serving
user_features = await store.serve_features_async(user_id=123)

# Batch feature retrieval
batch_features = await store.get_features_async(
    user_ids=[123, 456, 789],
    use_cache=True
)

# Get latest features for training
training_features = await store.get_features_async()
```

## 🏗 Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AdvancedFeatureStore                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Cache     │  │  Monitor    │  │  Quality Validator  │  │
│  │  Backend    │  │   System    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Feature    │  │  Database   │  │    Configuration    │  │
│  │ Extractors  │  │   Layer     │  │     Management      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Abstractions

- **`FeatureExtractor`**: Abstract base for feature engineering logic
- **`CacheBackend`**: Pluggable caching interface (Redis, Memory, etc.)
- **`DataQualityValidator`**: Configurable data validation framework
- **`FeatureMonitor`**: Observability and alerting system

## 📚 Usage Examples

### Custom Feature Extractor

```python
class CustomExtractor(FeatureExtractor):
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your custom feature logic
        return processed_features
    
    def get_feature_configs(self) -> List[FeatureConfig]:
        return [
            FeatureConfig(
                name="custom_feature",
                dtype="float64",
                description="Custom computed feature",
                tags=["custom"]
            )
        ]
```

### Data Quality Monitoring

```python
# Add custom quality checks
store.validator.add_check(
    lambda df: df['amount'].min() >= 0,
    "non_negative_amounts"
)

# Get quality metrics
metadata = await store.get_feature_metadata_async(feature_version)
print(f"Data quality score: {metadata.data_quality_score}")
```

### Advanced Querying

```python
# Get features for specific users with caching
features = await store.get_features_async(
    feature_version="abc123...",
    user_ids=[1, 2, 3],
    use_cache=True
)

# List all feature versions
versions = await store.list_feature_versions_async()
for version in versions:
    print(f"{version['feature_version']}: {version['description']}")
```

### Monitoring Dashboard

```python
# Get monitoring metrics
dashboard = store.get_monitoring_dashboard()
print("Feature Access Metrics:", dashboard['metrics'])
print("Active Alerts:", dashboard['alerts'])
```

## ⚙️ Configuration

### YAML Configuration

```yaml
# config/advanced_config.yml
raw_data: "data/raw_events.csv"
feature_store_db: "advanced_feature_store.db"
feature_table: "features"
feature_metadata_table: "feature_metadata"
cache_ttl: 3600
data_quality_threshold: 0.8
cleanup_schedule: "daily"

monitoring:
  enable_alerts: true
  alert_thresholds:
    quality_score: 0.8
    null_percentage: 0.1
```

### Environment Variables

```bash
export FEATURE_STORE_DB_PATH="/path/to/feature_store.db"
export CACHE_TTL=3600
export QUALITY_THRESHOLD=0.8
```

## 📊 Data Quality

### Quality Metrics

The system automatically calculates:

- **Null Percentage**: Ratio of missing values
- **Duplicate Percentage**: Ratio of duplicate rows
- **Outlier Percentage**: Statistical outliers using IQR method
- **Schema Violations**: Failed validation checks
- **Overall Score**: Composite quality score (0-1)

### Custom Quality Checks

```python
def check_user_id_format(df):
    return df['user_id'].dtype == 'int64'

store.validator.add_check(check_user_id_format, "valid_user_id_format")
```

### Quality Thresholds

```python
# Features with quality score < 0.8 trigger alerts
if quality_score < 0.8:
    logger.warning(f"Low quality features: {feature_version}")
```

## 📈 Monitoring

### Metrics Collected

- Feature access counts by version
- Feature creation timestamps and quality scores
- Cache hit/miss ratios
- Database query performance
- Data quality trends

### Alerting

- Low data quality scores
- High null percentages
- Schema validation failures
- Performance degradation

### Dashboard

```python
dashboard = store.get_monitoring_dashboard()
# Returns:
# {
#   "metrics": {...},
#   "alerts": [...],
#   "cache_info": "...",
#   "database_path": "..."
# }
```

## 📖 API Reference

### Core Methods

#### `register_features_async(features, metadata)`
Register new features with validation and monitoring.

**Parameters:**
- `features` (pd.DataFrame): Feature data
- `metadata` (FeatureMetadata): Feature metadata

**Returns:** Feature version hash (str)

#### `get_features_async(feature_version=None, user_ids=None, use_cache=True)`
Retrieve features with optional filtering and caching.

**Parameters:**
- `feature_version` (str, optional): Specific version to retrieve
- `user_ids` (List[int], optional): Filter by user IDs
- `use_cache` (bool): Enable/disable caching

**Returns:** Features DataFrame

#### `serve_features_async(user_id, feature_version=None)`
Serve features for real-time inference.

**Parameters:**
- `user_id` (int): Target user ID
- `feature_version` (str, optional): Specific version

**Returns:** Feature dictionary

### Data Models

#### `FeatureConfig`
```python
FeatureConfig(
    name="feature_name",
    dtype="float64",
    description="Feature description",
    tags=["tag1", "tag2"],
    owner="team_name"
)
```

#### `FeatureMetadata`
```python
FeatureMetadata(
    feature_version="abc123...",
    description="Feature set description",
    created_at="2024-01-01T00:00:00",
    features=[...],  # List of FeatureConfig
    data_quality_score=0.95,
    lineage={...},
    tags=[...]
)
```

## 🛡️ Best Practices

### Performance
- Use async methods for better concurrency
- Enable caching for frequently accessed features
- Batch user requests when possible
- Clean up old feature versions regularly

### Data Quality
- Set appropriate quality thresholds
- Add domain-specific validation checks
- Monitor quality scores over time
- Alert on quality degradation

### Monitoring
- Track feature access patterns
- Monitor cache performance
- Set up alerts for quality issues
- Review monitoring dashboard regularly

### Security
- Validate input data schemas
- Sanitize user inputs
- Use proper database connections
- Implement access controls as needed

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/test_feature_store.py
```

### Integration Tests
```bash
python -m pytest tests/test_integration.py
```

### Performance Tests
```bash
python -m pytest tests/test_performance.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `pytest`
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings for public methods
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [pandas](https://pandas.pydata.org/) for data manipulation
- Uses [asyncio](https://docs.python.org/3/library/asyncio.html) for async operations
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [SQLite](https://sqlite.org/) for reliable storage

## 📞 Support

- 📖 [Documentation](https://github.com/yourusername/advanced-feature-store/wiki)
- 🐛 [Issue Tracker](https://github.com/yourusername/advanced-feature-store/issues)
- 💬 [Discussions](https://github.com/yourusername/advanced-feature-store/discussions)
- 📧 Email: support@yourorganization.com

---

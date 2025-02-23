# Directory structure:
# blockchain_anomaly/
# ├── __init__.py
# ├── anomaly_detector.py
# ├── data_processor.py
# ├── models/
# │   ├── __init__.py
# │   ├── isolation_forest.py
# │   └── lstm_detector.py
# ├── utils/
# │   ├── __init__.py
# │   └── helpers.py
# ├── config.py
# ├── requirements.txt
# └── README.md

# anomaly_detector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
from .models.isolation_forest import IsolationForestDetector
from .models.lstm_detector import LSTMDetector
from .data_processor import DataProcessor
from .utils.helpers import setup_logging

class BlockchainAnomalyDetector:
    """Main class for detecting anomalies in blockchain transactions."""
    
    def __init__(self, config: Dict):
        """
        Initialize the anomaly detector with configuration settings.
        
        Args:
            config (Dict): Configuration dictionary containing model parameters
        """
        self.config = config
        self.logger = setup_logging(__name__, config['log_level'])
        self.data_processor = DataProcessor()
        
        # Initialize models
        self.isolation_forest = IsolationForestDetector(
            contamination=config['isolation_forest']['contamination']
        )
        self.lstm_detector = LSTMDetector(
            sequence_length=config['lstm']['sequence_length'],
            hidden_units=config['lstm']['hidden_units']
        )
        
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in blockchain transactions using multiple detection methods.
        
        Args:
            transactions (pd.DataFrame): DataFrame containing blockchain transactions
                Required columns: ['timestamp', 'amount', 'sender', 'receiver', 'gas_price']
                
        Returns:
            pd.DataFrame: Original DataFrame with additional anomaly detection columns
        """
        try:
            self.logger.info("Starting anomaly detection process")
            
            # Preprocess data
            processed_data = self.data_processor.preprocess_transactions(transactions)
            
            # Extract features
            features = self._extract_features(processed_data)
            
            # Run different detection methods
            anomaly_scores = self._run_detection_methods(features)
            
            # Combine results
            results = self._combine_detection_results(processed_data, anomaly_scores)
            
            self.logger.info("Completed anomaly detection process")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
            
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant features for anomaly detection."""
        features = pd.DataFrame()
        
        # Transaction amount features
        features['amount'] = data['amount']
        features['amount_log'] = np.log1p(data['amount'])
        features['gas_price'] = data['gas_price']
        
        # Temporal features
        features['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        
        # Wallet features
        features['sender_freq'] = data.groupby('sender')['sender'].transform('count')
        features['receiver_freq'] = data.groupby('receiver')['receiver'].transform('count')
        
        # Amount statistics
        features['amount_mean'] = data.groupby('sender')['amount'].transform('mean')
        features['amount_std'] = data.groupby('sender')['amount'].transform('std')
        features['amount_zscore'] = (data['amount'] - features['amount_mean']) / features['amount_std'].fillna(1)
        
        return features
        
    def _run_detection_methods(self, features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run multiple anomaly detection methods and return their scores."""
        anomaly_scores = {}
        
        # Isolation Forest detection
        anomaly_scores['isolation_forest'] = self.isolation_forest.detect(features)
        
        # LSTM detection
        sequence_features = self._prepare_sequences(features)
        anomaly_scores['lstm'] = self.lstm_detector.detect(sequence_features)
        
        # Statistical detection
        anomaly_scores['statistical'] = self._statistical_detection(features)
        
        return anomaly_scores
        
    def _statistical_detection(self, features: pd.DataFrame) -> np.ndarray:
        """Perform statistical anomaly detection."""
        # Z-score based detection
        amount_zscore = np.abs(features['amount_zscore'])
        
        # Time pattern detection
        unusual_hours = (features['hour'] >= 0) & (features['hour'] <= 4)
        
        # Frequency based detection
        sender_freq_zscore = np.abs(
            (features['sender_freq'] - features['sender_freq'].mean()) / features['sender_freq'].std()
        )
        
        # Combine scores
        statistical_scores = (amount_zscore + 
                            unusual_hours.astype(int) * 2 + 
                            sender_freq_zscore)
        
        return statistical_scores
        
    def _prepare_sequences(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare sequential data for LSTM detection."""
        sequence_length = self.config['lstm']['sequence_length']
        sequences = []
        
        for i in range(len(features) - sequence_length + 1):
            sequence = features.iloc[i:i + sequence_length].values
            sequences.append(sequence)
            
        return np.array(sequences)
        
    def _combine_detection_results(
        self, 
        original_data: pd.DataFrame, 
        anomaly_scores: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Combine results from different detection methods."""
        results = original_data.copy()
        
        # Add individual scores
        for method, scores in anomaly_scores.items():
            results[f'{method}_score'] = scores
            
        # Calculate combined anomaly score
        weights = self.config['detection_weights']
        combined_score = sum(
            scores * weights[method] 
            for method, scores in anomaly_scores.items()
        )
        
        results['anomaly_score'] = combined_score
        results['is_anomaly'] = combined_score > self.config['anomaly_threshold']
        
        # Add risk levels
        results['risk_level'] = pd.cut(
            results['anomaly_score'],
            bins=[-np.inf, 0.3, 0.6, 0.8, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return results

# models/isolation_forest.py
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
    def detect(self, features: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            features (pd.DataFrame): Feature DataFrame
            
        Returns:
            np.ndarray: Anomaly scores
        """
        # Fit and predict
        self.model.fit(features)
        
        # Convert predictions to anomaly scores
        scores = -self.model.score_samples(features)
        
        # Normalize scores to [0, 1]
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return normalized_scores

# models/lstm_detector.py
import tensorflow as tf
import numpy as np
from typing import Tuple

class LSTMDetector:
    """LSTM-based anomaly detector for sequential patterns."""
    
    def __init__(self, sequence_length: int, hidden_units: int):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.hidden_units,
                input_shape=(self.sequence_length, None)
            ),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def detect(self, sequences: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in sequences using LSTM.
        
        Args:
            sequences (np.ndarray): Array of sequences
            
        Returns:
            np.ndarray: Anomaly scores
        """
        # Make predictions
        predictions = self.model.predict(sequences)
        
        # Calculate reconstruction error as anomaly score
        reconstruction_error = np.mean(
            np.abs(sequences[:, -1, :] - predictions),
            axis=1
        )
        
        # Normalize scores
        normalized_scores = (reconstruction_error - reconstruction_error.min()) / (
            reconstruction_error.max() - reconstruction_error.min()
        )
        
        return normalized_scores

# config.py
DEFAULT_CONFIG = {
    'log_level': 'INFO',
    'isolation_forest': {
        'contamination': 0.1
    },
    'lstm': {
        'sequence_length': 10,
        'hidden_units': 64
    },
    'detection_weights': {
        'isolation_forest': 0.4,
        'lstm': 0.3,
        'statistical': 0.3
    },
    'anomaly_threshold': 0.7
}

# requirements.txt
"""
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
tensorflow>=2.6.0
python-dateutil>=2.8.2
logging>=0.5.1.2
"""

# README.md
"""
# Blockchain Anomaly Detection

This package provides comprehensive anomaly detection capabilities for blockchain transactions. It uses multiple detection methods including Isolation Forest, LSTM, and statistical analysis to identify suspicious patterns in blockchain data.

## Features:
- Multiple detection methods
- Real-time processing capability
- Configurable detection parameters
- Detailed anomaly scoring and risk levels
- Comprehensive logging and monitoring

## Installation:
```bash
pip install -r requirements.txt
```

## Usage:
```python
from blockchain_anomaly import BlockchainAnomalyDetector
import pandas as pd

# Initialize detector
config = DEFAULT_CONFIG  # Customize as needed
detector = BlockchainAnomalyDetector(config)

# Load your transaction data
transactions = pd.DataFrame({
    'timestamp': [...],
    'amount': [...],
    'sender': [...],
    'receiver': [...],
    'gas_price': [...]
})

# Detect anomalies
results = detector.detect_anomalies(transactions)

# Access results
suspicious_transactions = results[results['is_anomaly']]
print(f"Found {len(suspicious_transactions)} suspicious transactions")
```

## Configuration:
Customize the detection parameters in `config.py` based on your needs.

## Contributing:
Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License:
MIT License
"""
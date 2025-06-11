"""
Configuration settings for the password strength project.
"""

# Model configuration
MODEL_CONFIG = {
    'embedding_dim': 8,
    'lstm_units': 64,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 10,
    'batch_size': 2,
    'validation_split': 0.1,
    'test_size': 0.2,
    'random_state': 42,
    'verbose': 1
}

# Heuristic configuration
HEURISTIC_CONFIG = {
    'min_length': 8,
    'require_uppercase': True,
    'require_lowercase': True,
    'require_digits': True,
    'require_special': True,
    'check_common_patterns': True,
    'check_sequential': True,
    'check_repeated': True
}

# Breach check configuration
BREACH_CHECK_CONFIG = {
    'enable_breach_check': True,
    'api_url': 'https://api.pwnedpasswords.com/range/',
    'use_k_anonymity': True
}
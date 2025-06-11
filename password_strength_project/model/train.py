import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from password_strength_project.config import MODEL_CONFIG

def train_model(custom_data=None):
    """
    Train an LSTM model to classify password strength.
    
    Args:
        custom_data (dict, optional): Custom dataset to use instead of the default.
        
    Returns:
        tuple: (trained model, character to index mapping, maximum password length)
    """
    # Use provided data or default dataset
    if custom_data is None:
        data = {
            "password": [
                "123456", "password", "qwerty", "admin123", "letmein!",
                "P@ssw0rd2023", "Secure$456", "MyStr0ngPass!", "abc123",
                "welcome", "monkey", "sunshine", "princess", "football",
                "C0mpl3xP@$$w0rd", "Sup3rS3cur3P@ss!", "R@nd0m$tr1ng2023",
                "Th1sIsV3ryStr0ng!", "N0tE@syT0Gu3ss!"
            ],
            "strength": [
                "weak", "weak", "weak", "weak", "weak", 
                "strong", "strong", "strong", "weak",
                "weak", "weak", "weak", "weak", "weak",
                "strong", "strong", "strong", "strong", "strong"
            ]
        }
    else:
        data = custom_data
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['strength_encoded'] = label_encoder.fit_transform(df['strength'])
    
    # Prepare data
    X = df['password'].values
    y = df['strength_encoded'].values
    
    # Get maximum password length for padding
    max_length = max(len(p) for p in X)
    
    # Create character mapping
    char_set = set("".join(X))
    char_to_index = {char: idx + 1 for idx, char in enumerate(char_set)}
    
    # Convert passwords to sequences
    X = [password_to_sequence(p, char_to_index) for p in X]
    X = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in X])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG['test_size'], 
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Build model
    model = Sequential([
        Embedding(
            input_dim=len(char_to_index) + 1, 
            output_dim=MODEL_CONFIG['embedding_dim'], 
            input_length=max_length
        ),
        LSTM(MODEL_CONFIG['lstm_units'], return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=MODEL_CONFIG['optimizer'], 
        loss=MODEL_CONFIG['loss'], 
        metrics=MODEL_CONFIG['metrics']
    )
    
    # Train model
    model.fit(
        X_train, y_train, 
        epochs=MODEL_CONFIG['epochs'], 
        batch_size=MODEL_CONFIG['batch_size'], 
        validation_split=MODEL_CONFIG['validation_split'],
        verbose=MODEL_CONFIG['verbose']
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model, char_to_index, max_length

def password_to_sequence(password, char_to_index):
    """
    Convert a password string to a sequence of indices.
    
    Args:
        password (str): The password to convert
        char_to_index (dict): Mapping of characters to indices
        
    Returns:
        list: Sequence of indices
    """
    return [char_to_index.get(char, 0) for char in password]
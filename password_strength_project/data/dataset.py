import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class PasswordDataset:
    """
    Class to manage password datasets and preprocessing.
    """
    
    def __init__(self, data=None, file_path=None):
        """
        Initialize the dataset.
        
        Args:
            data (dict, optional): Dictionary with 'password' and 'strength' keys
            file_path (str, optional): Path to CSV file containing password data
        """
        if data is not None:
            self.df = pd.DataFrame(data)
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            # Default dataset if none provided
            self.df = pd.DataFrame({
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
            })
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['strength_encoded'] = self.label_encoder.fit_transform(self.df['strength'])
        
        # Calculate max length and character set
        self.max_length = max(len(p) for p in self.df['password'].values)
        self.char_set = set("".join(self.df['password'].values))
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(self.char_set)}
    
    def get_processed_data(self, test_size=0.2, random_state=42):
        """
        Get processed data ready for model training.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, char_to_index, max_length)
        """
        X = self.df['password'].values
        y = self.df['strength_encoded'].values
        
        # Convert passwords to sequences
        X_processed = [self._password_to_sequence(p) for p in X]
        X_processed = np.array([np.pad(seq, (0, self.max_length - len(seq)), 'constant') for seq in X_processed])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, self.char_to_index, self.max_length
    
    def _password_to_sequence(self, password):
        """
        Convert a password to a sequence of indices.
        
        Args:
            password (str): Password to convert
            
        Returns:
            list: Sequence of indices
        """
        return [self.char_to_index.get(char, 0) for char in password]
    
    def add_passwords(self, passwords, strengths):
        """
        Add new passwords to the dataset.
        
        Args:
            passwords (list): List of password strings
            strengths (list): List of strength labels
        """
        new_data = pd.DataFrame({
            "password": passwords,
            "strength": strengths
        })
        
        # Update dataset
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        
        # Re-encode labels
        self.df['strength_encoded'] = self.label_encoder.fit_transform(self.df['strength'])
        
        # Update max length and character set
        self.max_length = max(len(p) for p in self.df['password'].values)
        self.char_set = set("".join(self.df['password'].values))
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(self.char_set)}
    
    def save_to_csv(self, file_path):
        """
        Save the dataset to a CSV file.
        
        Args:
            file_path (str): Path to save the CSV file
        """
        self.df[['password', 'strength']].to_csv(file_path, index=False)
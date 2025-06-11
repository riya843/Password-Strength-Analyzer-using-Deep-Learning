import numpy as np

def evaluate_password_strength(model, char_to_index, max_length, password):
    """
    Evaluate the strength of a password using the trained model.
    
    Args:
        model: Trained Keras model
        char_to_index (dict): Character to index mapping
        max_length (int): Maximum password length for padding
        password (str): Password to evaluate
        
    Returns:
        tuple: (strength classification, probability)
    """
    # Convert password to sequence
    def password_to_sequence(password):
        return [char_to_index.get(char, 0) for char in password]
    
    # Prepare input
    seq = password_to_sequence(password)
    seq = np.pad(seq, (0, max_length - len(seq)), 'constant')
    seq = np.array([seq])
    
    # Get prediction
    strength_prob = model.predict(seq, verbose=0)[0][0]
    
    # Classify based on probability
    strength = "Strong" if strength_prob > 0.5 else "Weak"
    
    return strength, float(strength_prob)
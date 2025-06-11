import hashlib

# This is a simplified version. In a real application, you would use an API like
# Have I Been Pwned (HIBP) or a local database of breached passwords.
def check_password_breach(password):
    """
    Check if a password has been found in known data breaches.
    
    This is a simplified implementation. In a real application, you would:
    1. Use the Have I Been Pwned API (or similar service)
    2. Use k-anonymity to protect the full password by only sending a prefix of the hash
    
    Args:
        password (str): Password to check
        
    Returns:
        bool: True if password is found in breaches, False otherwise
    """
    # List of common breached password hashes (SHA-1)
    # In a real application, this would be a much larger database or API call
    common_breached_passwords = [
        "7c4a8d09ca3762af61e59520943dc26494f8941b",  # 123456
        "5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8",  # password
        "b1b3773a05c0ed0176787a4f1574ff0075f7521e",  # qwerty
        "e5e9fa1ba31ecd1ae84f75caaa474f3a663f05f4",  # secret
        "5ebe2294ecd0e0f08eab7690d2a6ee69a2138e0f",  # admin
        "f7c3bc1d808e04732adf679965ccc34ca7ae3441",  # admin123
        "ef92b778bafe771e89245b89ecbc08a44a4e166c",  # password123
        "b9c950640e1b3740e98acb93e669c65766f6670dd1609ba91ff41052ba48c6f3",  # letmein (SHA-256)
    ]
    
    # Calculate SHA-1 hash of the password
    sha1_hash = hashlib.sha1(password.encode()).hexdigest()
    
    # Check if the hash is in the list of breached passwords
    if sha1_hash in common_breached_passwords:
        return True
    
    # Also check SHA-256 for demonstration purposes
    sha256_hash = hashlib.sha256(password.encode()).hexdigest()
    if sha256_hash in common_breached_passwords:
        return True
    
    return False

def get_breach_details(password):
    """
    Get details about breaches where the password was found.
    
    In a real application, this would return information about which breaches
    contained the password, when they occurred, etc.
    
    Args:
        password (str): Password to check
        
    Returns:
        list: List of breach details (empty if no breaches)
    """
    is_breached = check_password_breach(password)
    
    if is_breached:
        # This is simulated data
        return [
            {
                "name": "Example Breach",
                "date": "2023-01-15",
                "records_affected": "10,000,000"
            }
        ]
    
    return []
import re

def check_password_heuristics(password):
    """
    Check password strength using rule-based heuristics.
    
    Args:
        password (str): Password to check
        
    Returns:
        dict: Dictionary of check results with pass/fail status and messages
    """
    results = {}
    
    # Check length
    min_length = 8
    results["length"] = {
        "passed": len(password) >= min_length,
        "message": f"Password should be at least {min_length} characters long."
    }
    
    # Check for uppercase letters
    has_uppercase = bool(re.search(r'[A-Z]', password))
    results["uppercase"] = {
        "passed": has_uppercase,
        "message": "Password should contain at least one uppercase letter."
    }
    
    # Check for lowercase letters
    has_lowercase = bool(re.search(r'[a-z]', password))
    results["lowercase"] = {
        "passed": has_lowercase,
        "message": "Password should contain at least one lowercase letter."
    }
    
    # Check for digits
    has_digit = bool(re.search(r'\d', password))
    results["digits"] = {
        "passed": has_digit,
        "message": "Password should contain at least one digit."
    }
    
    # Check for special characters
    has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    results["special_chars"] = {
        "passed": has_special,
        "message": "Password should contain at least one special character."
    }
    
    # Check for common patterns
    common_patterns = [
        r'123456', r'password', r'qwerty', r'abc123', r'admin',
        r'welcome', r'letmein', r'monkey', r'sunshine', r'princess'
    ]
    has_common_pattern = any(re.search(pattern, password.lower()) for pattern in common_patterns)
    results["common_patterns"] = {
        "passed": not has_common_pattern,
        "message": "Password should not contain common patterns or words."
    }
    
    # Check for sequential characters
    has_sequential_chars = bool(re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|012|123|234|345|456|567|678|789|890)', password.lower()))
    results["sequential_chars"] = {
        "passed": not has_sequential_chars,
        "message": "Password should not contain sequential characters."
    }
    
    # Check for repeated characters
    has_repeated_chars = bool(re.search(r'(.)\1{2,}', password))
    results["repeated_chars"] = {
        "passed": not has_repeated_chars,
        "message": "Password should not contain repeated characters."
    }
    
    return results
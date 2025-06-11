import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from password_strength_project.model.evaluate import evaluate_password_strength
from password_strength_project.model.train import train_model
from password_strength_project.utils.heuristics import check_password_heuristics
from password_strength_project.utils.breach_check import check_password_breach

def main():
    """
    Main entry point for the password strength checker application.
    """
    print("Password Strength Checker")
    print("========================")
    
    # Train the model first
    model, char_to_index, max_length = train_model()
    
    while True:
        password = input("\nEnter a password to check (or 'q' to quit): ")
        if password.lower() == 'q':
            break
        
        # Check password using ML model
        ml_strength, probability = evaluate_password_strength(model, char_to_index, max_length, password)
        
        # Check password using heuristics
        heuristic_results = check_password_heuristics(password)
        
        # Check if password has been breached
        breach_status = check_password_breach(password)
        
        # Display results
        print("\nPassword Strength Analysis:")
        print(f"ML Model Assessment: {ml_strength} (confidence: {probability:.2f})")
        print("\nHeuristic Checks:")
        for check, result in heuristic_results.items():
            status = "✓" if result["passed"] else "✗"
            print(f" {status} {check}: {result['message']}")
        
        print(f"\nBreach Check: {'Password found in data breaches!' if breach_status else 'No breaches found.'}")
        
        # Overall assessment
        if breach_status:
            overall = "VERY WEAK - Password has been compromised"
        elif ml_strength == "Weak" or sum(1 for r in heuristic_results.values() if not r["passed"]) > 1:
            overall = "WEAK - Consider a stronger password"
        else:
            overall = "STRONG - Good password choice"
        
        print(f"\nOVERALL ASSESSMENT: {overall}")

if __name__ == "__main__":
    main()
# Password Strength Checker

A comprehensive tool for evaluating password strength using machine learning and traditional heuristics.

## Features

- **Machine Learning Model**: LSTM-based neural network to classify password strength
- **Heuristic Checks**: Rule-based analysis of password characteristics
- **Breach Detection**: Check if a password has appeared in known data breaches
- **Comprehensive Analysis**: Combined assessment from multiple evaluation methods

## Project Structure

```
password_strength_project/
│── main.py                     # Entry point for running the project
│── model/
│   ├── train.py                # Train the LSTM model
│   ├── evaluate.py             # Evaluate password strength
│── data/
│   ├── dataset.py              # Manages password dataset & preprocessing
│── utils/
│   ├── heuristics.py           # Rule-based password checks
│   ├── breach_check.py         # Check against leaked passwords
│── config.py                   # Configurable hyperparameters
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r password_strength_project/requirements.txt
   ```

## Usage

Run the main script to start the password strength checker:

```
python -m password_strength_project.main
```

The application will:
1. Train a model on the default dataset
2. Prompt you to enter a password
3. Evaluate the password using multiple methods
4. Display a comprehensive analysis

## Customization

You can customize the behavior by modifying `config.py`:

- Adjust model hyperparameters
- Configure heuristic checks
- Enable/disable breach checking

## Extending the Project

- Add more sophisticated ML models
- Expand the training dataset
- Implement additional heuristic checks
- Connect to external breach databases

## License

MIT
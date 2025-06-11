import React, { useState, useEffect } from 'react';
import { Shield, ShieldAlert, ShieldCheck, AlertCircle, Check, X, Eye, EyeOff, Info } from 'lucide-react';

// Enhanced password strength checker with entropy calculation
const calculateEntropy = (password: string): number => {
  const charSet = new Set(password.split('')).size;
  return password.length * Math.log2(charSet || 1);
};

// Password strength checker functions
const checkLength = (password: string): boolean => password.length >= 8;
const checkUppercase = (password: string): boolean => /[A-Z]/.test(password);
const checkLowercase = (password: string): boolean => /[a-z]/.test(password);
const checkDigits = (password: string): boolean => /\d/.test(password);
const checkSpecialChars = (password: string): boolean => /[!@#$%^&*(),.?":{}|<>]/.test(password);
const checkCommonPatterns = (password: string): boolean => {
  const commonPatterns = [
    '123456', 'password', 'qwerty', 'abc123', 'admin',
    'welcome', 'letmein', 'monkey', 'sunshine', 'princess',
    'football', 'baseball', 'dragon', 'master', 'superman',
    'batman', 'trustno1', 'iloveyou', 'whatever', 'login'
  ];
  return !commonPatterns.some(pattern => password.toLowerCase().includes(pattern));
};
const checkSequentialChars = (password: string): boolean => {
  return !/abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz|012|123|234|345|456|567|678|789|890/i.test(password);
};
const checkRepeatedChars = (password: string): boolean => {
  return !/(.)(\1{2,})/.test(password);
};

// Expanded breach check with more common passwords
const checkBreached = (password: string): boolean => {
  const commonBreachedPasswords = [
    '123456', 'password', 'qwerty', 'secret', 'admin', 
    'admin123', 'password123', 'letmein', '12345678', 'baseball',
    'football', 'welcome', 'abc123', 'monkey', 'login',
    'princess', 'qwertyuiop', 'solo', 'passw0rd', 'starwars',
    'dragon', 'master', 'hello', 'freedom', 'whatever',
    'qazwsx', 'trustno1', 'iloveyou', 'sunshine', '1234567'
  ];
  return commonBreachedPasswords.includes(password.toLowerCase());
};

// Enhanced ML model simulation with feature engineering
const simulateMLModel = (password: string): number => {
  // Extract features
  const features = {
    length: password.length,
    uppercase: (password.match(/[A-Z]/g) || []).length,
    lowercase: (password.match(/[a-z]/g) || []).length,
    digits: (password.match(/\d/g) || []).length,
    specialChars: (password.match(/[!@#$%^&*(),.?":{}|<>]/g) || []).length,
    entropy: calculateEntropy(password)
  };
  
  // Calculate score based on features
  let score = 0;
  
  // Length contribution (up to 0.25)
  score += Math.min(features.length / 20, 0.25);
  
  // Character diversity contribution (up to 0.35)
  const charDiversity = (
    (features.uppercase > 0 ? 0.1 : 0) +
    (features.lowercase > 0 ? 0.1 : 0) +
    (features.digits > 0 ? 0.1 : 0) +
    (features.specialChars > 0 ? 0.05 : 0)
  );
  score += charDiversity;
  
  // Entropy contribution (up to 0.3)
  score += Math.min(features.entropy / 100, 0.3);
  
  // Pattern checks (up to 0.1)
  if (checkCommonPatterns(password)) score += 0.04;
  if (checkSequentialChars(password)) score += 0.03;
  if (checkRepeatedChars(password)) score += 0.03;
  
  return Math.min(score, 1);
};

// Classify password strength based on score and features
const classifyPasswordStrength = (password: string, score: number): string => {
  if (checkBreached(password)) {
    return "Very Weak - Compromised";
  }
  
  const entropy = calculateEntropy(password);
  const length = password.length;
  
  if (length < 6 || entropy < 20 || score < 0.2) {
    return "Very Weak";
  } else if (length < 8 || entropy < 30 || score < 0.4) {
    return "Weak";
  } else if (length < 10 || entropy < 40 || score < 0.6) {
    return "Moderate";
  } else if (length < 14 || entropy < 50 || score < 0.8) {
    return "Strong";
  } else {
    return "Very Strong";
  }
};

type CheckResult = {
  name: string;
  passed: boolean;
  message: string;
};

type PasswordFeatures = {
  length: number;
  uppercase: number;
  lowercase: number;
  digits: number;
  specialChars: number;
  entropy: number;
};

function App() {
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [checks, setChecks] = useState<CheckResult[]>([]);
  const [mlScore, setMlScore] = useState(0);
  const [isBreached, setIsBreached] = useState(false);
  const [overallStrength, setOverallStrength] = useState('');
  const [strengthColor, setStrengthColor] = useState('');
  const [features, setFeatures] = useState<PasswordFeatures | null>(null);
  const [showFeatures, setShowFeatures] = useState(false);

  useEffect(() => {
    if (password) {
      // Extract features
      const extractedFeatures: PasswordFeatures = {
        length: password.length,
        uppercase: (password.match(/[A-Z]/g) || []).length,
        lowercase: (password.match(/[a-z]/g) || []).length,
        digits: (password.match(/\d/g) || []).length,
        specialChars: (password.match(/[!@#$%^&*(),.?":{}|<>]/g) || []).length,
        entropy: calculateEntropy(password)
      };
      
      setFeatures(extractedFeatures);
      
      // Run all checks
      const results: CheckResult[] = [
        {
          name: 'Length',
          passed: checkLength(password),
          message: 'Password should be at least 8 characters long'
        },
        {
          name: 'Uppercase',
          passed: checkUppercase(password),
          message: 'Password should contain at least one uppercase letter'
        },
        {
          name: 'Lowercase',
          passed: checkLowercase(password),
          message: 'Password should contain at least one lowercase letter'
        },
        {
          name: 'Digits',
          passed: checkDigits(password),
          message: 'Password should contain at least one digit'
        },
        {
          name: 'Special Characters',
          passed: checkSpecialChars(password),
          message: 'Password should contain at least one special character'
        },
        {
          name: 'Common Patterns',
          passed: checkCommonPatterns(password),
          message: 'Password should not contain common patterns or words'
        },
        {
          name: 'Sequential Characters',
          passed: checkSequentialChars(password),
          message: 'Password should not contain sequential characters'
        },
        {
          name: 'Repeated Characters',
          passed: checkRepeatedChars(password),
          message: 'Password should not contain repeated characters'
        }
      ];
      
      setChecks(results);
      
      // Check if breached
      const breached = checkBreached(password);
      setIsBreached(breached);
      
      // Get ML score
      const score = simulateMLModel(password);
      setMlScore(score);
      
      // Determine overall strength
      const strength = classifyPasswordStrength(password, score);
      setOverallStrength(strength);
      
      // Set color based on strength
      let color = '';
      if (strength.includes('Very Weak')) {
        color = 'text-red-600';
      } else if (strength.includes('Weak')) {
        color = 'text-orange-500';
      } else if (strength.includes('Moderate')) {
        color = 'text-yellow-500';
      } else if (strength.includes('Strong')) {
        color = 'text-green-600';
      } else {
        color = 'text-emerald-600';
      }
      setStrengthColor(color);
    } else {
      // Reset everything if password is empty
      setChecks([]);
      setMlScore(0);
      setIsBreached(false);
      setOverallStrength('');
      setFeatures(null);
    }
  }, [password]);

  const getStrengthIcon = () => {
    if (!password) return null;
    
    if (isBreached || overallStrength.includes('Very Weak')) {
      return <ShieldAlert className="w-8 h-8 text-red-600" />;
    }
    
    if (overallStrength.includes('Weak')) {
      return <ShieldAlert className="w-8 h-8 text-orange-500" />;
    } else if (overallStrength.includes('Moderate')) {
      return <Shield className="w-8 h-8 text-yellow-500" />;
    } else if (overallStrength.includes('Strong')) {
      return <ShieldCheck className="w-8 h-8 text-green-600" />;
    } else {
      return <ShieldCheck className="w-8 h-8 text-emerald-600" />;
    }
  };

  const getProgressBarColor = () => {
    if (!password) return 'bg-gray-200';
    if (isBreached || overallStrength.includes('Very Weak')) return 'bg-red-600';
    if (overallStrength.includes('Weak')) return 'bg-orange-500';
    if (overallStrength.includes('Moderate')) return 'bg-yellow-500';
    if (overallStrength.includes('Strong')) return 'bg-green-600';
    return 'bg-emerald-600';
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl">
        <div className="p-8">
          <div className="flex items-center justify-center mb-6">
            <Shield className="h-10 w-10 text-indigo-600 mr-2" />
            <h1 className="text-2xl font-bold text-gray-900">Password Strength Checker</h1>
          </div>
          
          <div className="mb-6">
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
              Enter a password to check
            </label>
            <div className="relative rounded-md shadow-sm">
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pr-10 sm:text-sm border-gray-300 rounded-md p-2 border"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <button
                type="button"
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? (
                  <EyeOff className="h-5 w-5 text-gray-400" />
                ) : (
                  <Eye className="h-5 w-5 text-gray-400" />
                )}
              </button>
            </div>
          </div>
          
          {password && (
            <>
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Strength:</span>
                  <span className={`text-sm font-semibold ${strengthColor}`}>{overallStrength}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full ${getProgressBarColor()}`} 
                    style={{ width: `${mlScore * 100}%` }}
                  ></div>
                </div>
              </div>
              
              {isBreached && (
                <div className="mb-6 p-3 bg-red-100 border border-red-200 rounded-md flex items-start">
                  <AlertCircle className="w-5 h-5 text-red-600 mr-2 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-sm font-medium text-red-800">Password Breach Detected!</h3>
                    <p className="text-sm text-red-700 mt-1">
                      This password has been found in known data breaches. Using it puts your account at serious risk.
                    </p>
                  </div>
                </div>
              )}
              
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Password Requirements:</h3>
                <div className="space-y-2">
                  {checks.map((check, index) => (
                    <div key={index} className="flex items-start">
                      {check.passed ? (
                        <Check className="w-5 h-5 text-green-600 mr-2 flex-shrink-0 mt-0.5" />
                      ) : (
                        <X className="w-5 h-5 text-red-600 mr-2 flex-shrink-0 mt-0.5" />
                      )}
                      <span className={`text-sm ${check.passed ? 'text-gray-600' : 'text-red-600'}`}>
                        {check.message}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="mb-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-gray-700">Password Analysis:</h3>
                  <button 
                    onClick={() => setShowFeatures(!showFeatures)}
                    className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center"
                  >
                    {showFeatures ? 'Hide Details' : 'Show Details'}
                    <Info className="h-4 w-4 ml-1" />
                  </button>
                </div>
                
                {showFeatures && features && (
                  <div className="mt-3 p-3 bg-gray-50 rounded-md">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Length:</span> {features.length}
                      </div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Entropy:</span> {features.entropy.toFixed(1)}
                      </div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Uppercase:</span> {features.uppercase}
                      </div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Lowercase:</span> {features.lowercase}
                      </div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Digits:</span> {features.digits}
                      </div>
                      <div className="text-sm">
                        <span className="font-medium text-gray-700">Special:</span> {features.specialChars}
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="bg-gray-50 p-4 rounded-md">
                <div className="flex items-center">
                  {getStrengthIcon()}
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-gray-700">ML Model Assessment</h3>
                    <p className="text-sm text-gray-500">
                      Confidence score: {(mlScore * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
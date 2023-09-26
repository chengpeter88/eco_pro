import numpy as np

def ces_constructor(alpha, sigma):
    def ces_utility(X):
        if not isinstance(alpha, (float, int, np.ndarray)):
            raise ValueError("Alpha must be a float, int, or numpy array.")
        
        if isinstance(alpha, (float, int)):
            alpha_array = np.full(X.shape, alpha)
        else:
            alpha_array = alpha
        
        return np.sum(alpha_array * X**(-sigma))**(-1/sigma)
    
    return ces_utility

# Create a CES utility function with alpha=0.7 and sigma=0.5
ces = ces_constructor(0.7, 0.5)

# Test the CES utility function with X as a numpy array
X = np.array([1.0, 2.0, 3.0])
result = ces(X)
print("CES Utility Function Result:", result)



import numpy as np
def ces_constructor(alpha, sigma):
    def ces_utility(X):
        if isinstance(alpha, (float, int)):
            alpha_array = np.full(X.shape, alpha)
        else:
            alpha_array = alpha
        return np.sum((X**(-sigma)) * alpha_array)**(-1/sigma)
    return ces_utility


ces = ces_constructor(0.7, 0.5)
X = np.array([1, 2, 3])
ces_value = ces(X)
print("CES Utility Value:", ces_value)

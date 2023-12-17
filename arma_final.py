import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.optimize import minimize
#################### ARMA class ####################
class ARMA:
    ### Q:if p,q 的長度不等於 phi,theta 的長度，會報錯
    def __init__(self, p, q, phi, theta, c, mu,sigma):
        if not isinstance(p, int) or not isinstance(q, int):
            raise ValueError("p and q must be integers")
        if len(phi) != p:
            raise ValueError("Length of phi must be equal to p")
        if len(theta) != q:
            raise ValueError("Length of theta must be equal to q")
        self.p = p  # Order of the AR component
        self.q = q  # Order of the MA component
        self.phi = np.array(phi)  # AR coefficients
        self.theta = np.array(theta)  # MA coefficients
        self.c = c  # Constant term
        self.sigma = sigma  # Standard deviation of the noise
        self.mu = mu  #mean  of the noise
        self.memory = np.zeros(max(p, q))  # Initialize memory for AR and MA parts
        self.noise_memory = np.zeros(q)  # Memory for noise terms in MA part
   
    def simulate(self, n=1):
        simulate(self, n) 

    def statistics(self):
        return statistics(self)

    def plot(self):
        plot(self, self.p, self.q)
    
    def save(self, file_name):
        save(self, file_name)

    def fit(self, data):
        fit(self, data)





#################### ARMA hepler function ####################
  
def simulate(arma, n=1):
    X = np.zeros(n)
    for t in range(n):
        noise = np.random.normal(arma.mu, arma.sigma)
            # AR part
        X[t] = arma.c + noise
        for i in range(min(t, arma.p)):
            X[t] += arma.phi[i] * arma.memory[i]
            # MA part
        for j in range(min(t+1, arma.q)):
            X[t] += arma.theta[j] * arma.noise_memory[j]
        arma.memory = np.concatenate(([X[t]], arma.memory))
        arma.noise_memory = np.concatenate(([noise], arma.noise_memory))
    return X

def statistics(arma):
    mean = arma.c / (1 - np.sum(arma.phi))
    variance = arma.sigma**2 * (1 + np.sum(arma.theta**2) + 2 * np.sum(arma.phi**2))
    return f"平均數:{mean:.4f}, 變異數:{variance:.4f}"
def plot(arma, p, q):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.plot(arma.memory, marker='o')
    plt.title(f"ARMA({p}, {q}) Process")
    plt.xlabel("period")
    plt.ylabel("Value")
    plt.show()

def save(arma, file_name):
    df = pd.DataFrame(arma.memory)
    if file_name.endswith('.xlsx'):
        df.to_excel(file_name, index=False)
    elif file_name.endswith('.txt'):
        df.to_csv(file_name, index=False, sep='\t')
    elif file_name.endswith('.csv'):
        df.to_csv(file_name, index=False)
    else:
        raise ValueError("file_name must end with .xlsx or .txt or .csv")
    
def fit(arma, data):
        # Define the objective function for the optimizer
    def objective(params):
        arma.phi,arma.theta, arma.c, arma.mu, arma.sigma = params[:arma.p], params[arma.p:arma.p+arma.q], params[-3], params[-2], params[-1]
        simulated_data = arma.simulate(len(data))
        return np.sum((data - simulated_data)**2)

        # Use an optimizer to find the parameters that minimize the objective function
    initial_guess = np.concatenate([arma.phi, arma.theta, [arma.c, arma.mu, arma.sigma]])
    result = minimize(objective, initial_guess)
    arma.phi, arma.theta, arma.c, arma.mu, arma.sigma = result.x[:arma.p], result.x[arma.p:arma.p+arma.q], result.x[-3], result.x[-2], result.x[-1]
    
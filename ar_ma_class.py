import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
###ar= AR(0.5, epsilon=lambda n: np.random.normal(0, 1, n), Y0=[0])
###ar.simulate_nPeriods(100)
###ar.statistics()
#  
#################### ARMA class ####################
class AR:
    def __init__(self, *args, epsilon, Y0):
        self.phi = np.array(args)
        self.epsilon = epsilon
        self.Y0 = np.array(Y0)
        self.memory = np.array([])
        self.YPast = np.array(self.Y0)
    def simulate_nPeriods(self, n=1):
        simulate_nPeriods(self, n)
    
    #def clear_memory(self):
    #    self.memory = np.array([])
    
    def statistics(self):
        return statistics(self)
    
    def plot(self):
        plot(self)
    
    def save(self, file_name):
        save(self, file_name)
#################### MA class ####################
class MA:
    def __init__(self, q, theta, mu, sigma):
        if not isinstance(q, int) :
            raise ValueError(" q must be integers")
        if len(theta) != q:
            raise ValueError("Length of theta must be equal to q")
    
        self.q = q  # MA model order
        self.theta = np.array(theta)  # MA coefficients
        self.mu = mu  # Mean of the series
        self.sigma = sigma  # Standard deviation of the noise
        self.memory = np.zeros(q)  # Initialize memory to store past noise values

        # """條件需要符合 q 是整數，否則會報錯，需要有套件"""
    
    def simulate(self, n=1):
        """
        預設參數為模擬一期，可以自行設定調用模擬期數
        """
        return simulate(self, n)
    
    def statistics(self):
        return statistics(self)
    
    def covariance(self, k):
        if isinstance(k, int):
            return covariance(self, k)
        else:
            raise ValueError("k must be an integer")

    def plot(self):
        plot(self, self.q) 
    
    def save(self, file_name):
        save(self, file_name)  




#################### helpers  fuctions ####################
# AR helpers    

def simulate_onePeriod(ar, eps):
    y_onePeriod_ahead = ar.phi@ar.YPast + eps
    ar.YPast = np.append(y_onePeriod_ahead, ar.YPast[:-1])
    ar.memory = np.append(ar.memory, y_onePeriod_ahead)

def simulate_nPeriods(ar, n):
    eps = ar.epsilon(n)
    for i in range(n):
        simulate_onePeriod(ar, eps[i])

def statistics(ar):
    mu_val = np.mean(ar.memory)
    sigma_val = np.std(ar.memory)
    corr_val = np.corrcoef(ar.memory[1:], ar.memory[:-1])[0,1]  ## 2X2 矩陣[0,1]位子是相關係數  
    return f"平均數:{mu_val:.4f}, 變異數:{sigma_val:.4f}, 相關係數:{corr_val:.4f}"

def plot(ar):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.plot(ar.memory,marker='o')
    plt.title("AR(1) process")
    plt.xlabel("preiod")
    plt.ylabel("y")
    plt.show()

def save(ar, file_name):
    df = pd.DataFrame(ar.memory)
    if file_name.endswith('.xlsx'):
        df.to_excel(file_name, index=False)
    elif file_name.endswith('.txt'):
        df.to_csv(file_name, index=False, sep='\t')
    elif file_name.endswith('.csv'):
        df.to_csv(file_name, index=False)
    else:
        raise ValueError("file_name must end with .xlsx or .txt or .csv")

# MA helpers    

def simulate(self, n=1):
    noise = np.random.normal(0, self.sigma, n + self.q)  # Generate noise   
    X = np.zeros(n)
    for t in range(n):
        X[t] = self.mu + noise[t + self.q]  # Current noise term
        for i in range(min(self.q, t+1)):
             X[t] += self.theta[i] * noise[t + self.q - i - 1]  # Past noise terms
        self.memory = np.append(self.memory, X[t])  # Update memory with new value
    return X

def plot(ma, q):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.plot(ma.memory, marker='o')
    plt.title(f"MA({q}) Process")  
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

def statistics(ma):
    mean = ma.mu
    variance = ma.sigma**2 * (1 + np.sum(ma.theta**2))
    return f"平均數:{mean:.4f}, 變異數:{variance:.4f}"

def covariance(ma, k):
    for k in range(1,k+1):
        if k > ma.q:
            covariance = 0
        else:
            theta_padded = np.append(ma.theta, np.zeros(k))  # Pad theta with zeros for lag
            covariance = ma.sigma**2 * np.sum(theta_padded[k:ma.q+k] * ma.theta[:ma.q])
        print(f"共變異數（lag {k}）: {covariance:.4f}")

def save(ma, file_name):
    df = pd.DataFrame(ma.memory)
    if file_name.endswith('.xlsx'):
        df.to_excel(file_name, index=False)
    elif file_name.endswith('.txt'):
        df.to_csv(file_name, index=False, sep='\t')
    elif file_name.endswith('.csv'):
        df.to_csv(file_name, index=False)
    else:
        raise ValueError("file_name must end with .xlsx or .txt or .csv")


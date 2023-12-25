import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ARMA:
    def __init__(self, p:int, q:int, phi:list, theta:list, c:float, mu:float, sigma:float):
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
        self.mu = mu  # Mean of the noise
        self.memory = np.zeros(max(p, q))  # Initialize memory for AR and MA parts
        self.noise_memory = np.zeros(q)  # Memory for noise terms in MA part

    def simulate(self, num_samples=1):
        return self._simulate(num_samples)

    def statistics(self):
        return self._statistics()

    def plot(self):
        return self._plot()

    def save(self, file_name):
        return self._save(file_name)
    

    
    ##priveate function 
    def _simulate(self, num_samples):
        X = np.zeros(num_samples)
        for t in range(num_samples):
            noise = np.random.normal(self.mu, self.sigma)
            # AR part
            X[t] = self.c + noise
            for i in range(min(t, self.p)):
                X[t] += self.phi[i] * self.memory[i]
            # MA part
            for j in range(min(t+1, self.q)):
                X[t] += self.theta[j] * self.noise_memory[j]
            self.memory = np.concatenate(([X[t]], self.memory))
            self.noise_memory = np.concatenate(([noise], self.noise_memory))
        return X

    def _statistics(self):
        mean = self.c / (1 - np.sum(self.phi))
        variance = self.sigma**2 * (1 + np.sum(self.theta**2) + 2 * np.sum(self.phi**2))
        return f"Mean: {mean:.4f}, Variance: {variance:.4f}"

    def _plot(self):
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 4))
        plt.plot(self.memory, marker='o')
        plt.title(f"ARMA({self.p}, {self.q}) Process")
        plt.xlabel("Period")
        plt.ylabel("Value")
        plt.show()

    def _save(self, file_name):
        df = pd.DataFrame(self.memory)
        if file_name.endswith('.xlsx'):
            df.to_excel(file_name, index=False)
        elif file_name.endswith('.txt'):
            df.to_csv(file_name, index=False, sep='\t')
        elif file_name.endswith('.csv'):
            df.to_csv(file_name, index=False)
        else:
            raise ValueError("file_name must end with .xlsx or .txt or .csv")
        return f"File saved as {file_name}"
    

class AR(ARMA):
        def __init__(self, p, phi, c, mu, sigma):
            super().__init__(p, 0, phi, [], c, mu, sigma)  # 調用父類的構造方法，MA部分設為空
        def simulate(self, num_samples=1):
            return self._simulate(num_samples)
        def statistics(self):
            return self._statistics()
        def plot(self):
            return self._plot()
        def save(self, file_name):
            return self._save(file_name)
        



class MA(ARMA):
    def __init__(self, q, theta, mu, c,sigma):
        super().__init__(0, q, [], theta, c, mu, sigma)
    def simulate(self, num_samples=1):
        return self._simulate(num_samples)
    def statistics(self):
        return self._statistics()
    def plot(self):
        return self._plot()
    def save(self, file_name):
        return self._save(file_name)
    
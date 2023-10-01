class Demand:
    def __init__(self, ad, bd):
        self.ad = ad
        self.bd = bd

    def quantity(self, price):
        return self.ad - self.bd * price
    
    def __str__(self):
        return f'Demand(ad={self.ad}, bd={self.bd})'

class Supply:
    def __init__(self, az, bz):
        self.az = az
        self.bz = bz

    def quantity(self, price):
        "Compute quantity supplied at given price"
        return self.az + self.bz * price

    def __str__(self):
        return f'Supply(az={self.az}, bz={self.bz})'


class Market:
    def __init__(self, demand, supply):
        self.demand = demand
        self.supply = supply

    def price(self):
        "Compute equilibrium price"
        return (self.demand.ad - self.supply.az) / (self.demand.bd + self.supply.bz)

    def quantity(self):
        "Compute equilibrium quantity"
        return self.demand.quantity(self.price())

    def __str__(self):
        return f'Market(demand={self.demand}, supply={self.supply})'


# Create demand and supply curves
demand = Demand(ad=100, bd=0.5)
supply = Supply(az=20, bz=0.3)

# Create market instance
market = Market(demand=demand, supply=supply)

# Compute equilibrium price and quantity
price = market.price()
quantity = market.quantity()

# Print results
print(f'Equilibrium price: {price:.2f}')
print(f'Equilibrium quantity: {quantity:.2f}')
import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
n_simulations = 10000

# Simulate rolling a dice three times and summing the results, for n_simulations times
sums = np.sum(np.random.randint(1, 7, size=(n_simulations, 3)), axis=1)

# Plot the histogram of sums
plt.hist(sums, bins=range(3, 19), density=True, alpha=0.75, align='left')
plt.title('Probability Distribution of the Sum of Rolling a Dice 3 Times')
plt.xlabel('Sum')
plt.ylabel('Probability')
plt.xticks(range(3, 19))
plt.grid(axis='y', alpha=0.75)

plt.show()
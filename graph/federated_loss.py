import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('federated_loss.csv')

plt.plot(data['Round'], data['Loss'], marker='o', color='green')

plt.xlabel('# Rounds')
plt.ylabel('Loss')
plt.title('Loss due to federated learning')
plt.savefig('federated_loss_graph.png', format='png')
plt.show()

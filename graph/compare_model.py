import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('compare_model.csv')

plt.figure(figsize=(10, 6))
plt.bar(data['Model'], data['Loss'], color='skyblue')

plt.title('Compare Global and Client train loss')
plt.xlabel('Model from')
plt.ylabel('Loss')

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('compare_model_loss_graph.png', format='png')
plt.show()

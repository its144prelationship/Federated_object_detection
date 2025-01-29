import re
import matplotlib.pyplot as plt

def parse_log(log_file_path):
    rounds, losses, maps, recalls = [], [], [], []
    current_round, current_loss, current_map, current_recall = None, None, None, None
    
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            round_match = re.search(r"INFO:aggregation:### Round (\d+) ###", line)
            if round_match:
                if current_round is not None and current_loss is not None and current_map is not None and current_recall is not None:
                    rounds.append(current_round)
                    losses.append(current_loss)
                    maps.append(current_map)
                    recalls.append(current_recall)
                current_round = int(round_match.group(1))
                current_loss, current_map, current_recall = None, None, None
            
            loss_match = re.search(r"INFO:aggregation:server_test_loss ([\d.]+)", line)
            if loss_match:
                current_loss = float(loss_match.group(1))
            
            map_match = re.search(r"INFO:aggregation:server_test_map ([\d.]+)", line)
            if map_match:
                current_map = float(map_match.group(1))
            
            recall_match = re.search(r"INFO:aggregation:server_test_recall ([\d.]+)", line)
            if recall_match:
                current_recall = float(recall_match.group(1))
    
    if current_round is not None and current_loss is not None and current_map is not None and current_recall is not None:
        rounds.append(current_round)
        losses.append(current_loss)
        maps.append(current_map)
        recalls.append(current_recall)
    
    return rounds[:200], losses[:200], maps[:200], recalls[:200]

# Log file paths
log_file_1 = "0122-gpu.log"
log_file_2 = "1225.log"

# Parse logs
rounds_1, losses_1, maps_1, recalls_1 = parse_log(log_file_1)
rounds_2, losses_2, maps_2, recalls_2 = parse_log(log_file_2)

# Plot the graphs
plt.figure(figsize=(12, 6))

# Loss vs Round
plt.subplot(1, 3, 1)
plt.plot(rounds_1, losses_1, marker='o', label=f"Loss ({log_file_1})")
plt.plot(rounds_2, losses_2, marker='s', label=f"Loss ({log_file_2})")
plt.title("Round vs Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# MAP vs Round
plt.subplot(1, 3, 2)
plt.plot(rounds_1, maps_1, marker='o', label=f"mAP ({log_file_1})", color="orange")
plt.plot(rounds_2, maps_2, marker='s', label=f"mAP ({log_file_2})", color="red")
plt.title("Round vs mAP")
plt.xlabel("Round")
plt.ylabel("mAP")
plt.legend()
plt.grid(True)

# Recall vs Round
plt.subplot(1, 3, 3)
plt.plot(rounds_1, recalls_1, marker='o', label=f"Recall ({log_file_1})", color="green")
plt.plot(rounds_2, recalls_2, marker='s', label=f"Recall ({log_file_2})", color="blue")
plt.title("Round vs Recall")
plt.xlabel("Round")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"fl_comparison_{log_file_1.split('.')[0]}_and_{log_file_2.split('.')[0]}.png", format='png')
plt.show()

import re
import matplotlib.pyplot as plt

log_file_path = "1216.log"

rounds = []
losses = []
maps = []
recalls = []

current_round = None
current_loss = None
current_map = None
current_recall = None

with open(log_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        round_match = re.search(r"INFO:aggregation:### Round (\d+) ###", line)
        if round_match:
            # If a new round starts, finalize the previous round's data
            if current_round is not None and current_loss is not None and current_map is not None and current_recall is not None:
                rounds.append(current_round)
                losses.append(current_loss)
                maps.append(current_map)
                recalls.append(current_recall)
            # Start a new round
            current_round = int(round_match.group(1))
            current_loss = None
            current_map = None
            current_recall = None
        
        loss_match = re.search(r"INFO:aggregation:server_test_loss ([\d.]+)", line)
        if loss_match:
            current_loss = float(loss_match.group(1))
        
        map_match = re.search(r"INFO:aggregation:server_test_map ([\d.]+)", line)
        if map_match:
            current_map = float(map_match.group(1))
        
        recall_match = re.search(r"INFO:aggregation:server_test_recall ([\d.]+)", line)
        if recall_match:
            current_recall = float(recall_match.group(1))

    # Finalize the last round
    if current_round is not None and current_loss is not None and current_map is not None and current_recall is not None:
        rounds.append(current_round)
        losses.append(current_loss)
        maps.append(current_map)
        recalls.append(current_recall)


# Plot the graphs
plt.figure(figsize=(12, 6))

# Loss vs Round
plt.subplot(1, 3, 1)
plt.plot(rounds, losses, marker='o', label="Loss")
plt.title("Round vs Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

# MAP vs Round
plt.subplot(1, 3, 2)
plt.plot(rounds, maps, marker='o', label="mAP", color="orange")
plt.title("Round vs mAP")
plt.xlabel("Round")
plt.ylabel("mAP")
plt.grid(True)

# Recall vs Round
plt.subplot(1, 3, 3)
plt.plot(rounds, recalls, marker='o', label="Recall", color="green")
plt.title("Round vs Recall")
plt.xlabel("Round")
plt.ylabel("Recall")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"fl_server_{log_file_path.split('.')[0]}.png", format='png')
plt.show()
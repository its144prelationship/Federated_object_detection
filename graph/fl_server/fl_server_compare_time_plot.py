from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Define log files
log_files = ["1225-timestamp.log", "0122-timestamp.log"]  # Update with actual file paths
labels = ["CPU", "GPU"]  # Update labels accordingly
colors = ["orange", "darkcyan"]  # Colors for plots

def extract_round_times(log_file):
    round_times = []
    start_time = None

    with open(log_file, "r") as f:
        for line in f:
            if "### Round" in line:
                timestamp_str = line.split(" - ")[0]  # Extract timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

                if start_time is not None:
                    round_time = (timestamp - start_time).total_seconds()
                    round_times.append(round_time)

                start_time = timestamp  # Update start time for next round

    return round_times[:50]  # Keep only last 200 rounds

# Extract times for both files
round_times_1 = extract_round_times(log_files[0])
round_times_2 = extract_round_times(log_files[1])

# Compute averages
avg_time_1 = np.mean(round_times_1)
avg_time_2 = np.mean(round_times_2)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(round_times_1) + 1), round_times_1, marker='o', color=colors[0], label=f"{labels[0]} (Avg: {avg_time_1:.2f} sec)")
plt.plot(range(1, len(round_times_2) + 1), round_times_2, marker='s', color=colors[1], label=f"{labels[1]} (Avg: {avg_time_2:.2f} sec)")
plt.axhline(y=avg_time_1, color=colors[0], linestyle="--")
plt.axhline(y=avg_time_2, color=colors[1], linestyle="--")

plt.xlabel("Round Number")
plt.ylabel("Time Taken (seconds)")
plt.title("Comparison of Round Time")
plt.legend()
plt.grid()
plt.savefig(f"fl_server_time_compare_btw_{log_files[0].split('-')[0]}_and_{log_files[1].split('-')[0]}.png", format='png')
plt.show()

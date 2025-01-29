from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

log_file = "1225-timestamp.log"  # Change to your actual log file
round_times = []
start_time = None

with open(log_file, "r") as f:
    for line in f:
        if "### Round" in line:
            timestamp_str = line.split(" - ")[0]  # Extract "YYYY-MM-DD HH:MM:SS,mmm"
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            if start_time is not None:
                round_time = (timestamp - start_time).total_seconds()
                round_times.append(round_time)

            start_time = timestamp  # Update start time for next round

# Keep only the last 200 rounds
round_times = round_times[-200:]

# Compute average round time
avg_round_time = np.mean(round_times)

# Plot the time taken per round
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(round_times) + 1), round_times, marker='o', label="Round Time")
plt.axhline(y=avg_round_time, color='r', linestyle='--', label=f"Avg: {avg_round_time:.2f} sec")
plt.xlabel("Round Number")
plt.ylabel("Time Taken (seconds)")
plt.title("Time Used in Each Round (Last 200 Rounds)")
plt.legend()
plt.grid()
plt.show()

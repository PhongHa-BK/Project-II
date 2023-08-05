import time
import os.path as osp

log_file_path = ""

def write_log(log_file_path, content):
    with open(log_file_path, 'a') as log_file:
        log_file.write(content + '\n')

# Intialize the array to save training time per each epoch
epoch_times_list = []

# Starting to calculate training model time
start_time = time.time()

# Estimate the training time of each epoch
epoch_start_time = time.time()
end_time = time.time()
elapsed_time = end_time - epoch_start_time
elapsed_time_str = f"{int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s"
epoch_times_list.append((epoch + 1, elapsed_time))

# Calculate total training time
total_elapsed_time = sum([epoch_times[1] for epoch_times in epoch_times_list])
total_elapsed_time_str = f"{int(total_elapsed_time // 3600)}h {int((total_elapsed_time % 3600) // 60)}m {int(total_elapsed_time % 60)}s"
write_log(log_file_path, f"\tElapsed Time: {elapsed_time_str}")
write_log(log_file_path, f"\tTotal Elapsed Time: {total_elapsed_time_str}\n{'-'*50}")
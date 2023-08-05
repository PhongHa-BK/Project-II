log_file_path = ""
thread = 3 # thread can be change to fit the number of epoch
# definition a function to write data to log file
def write_log(log_file_path, content):
    with open(log_file_path, 'a') as log_file:
        log_file.write(content + '\n')

# definition a function to read data from log file
def read_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    # Check if log file contains data or being empty
    if len(lines) == 0:
        return 0, None

    # Get the number of epochs has been completed
    num_epochs_completed = len(lines) 

    # Check if the remained lines of the log file
    if len(lines) < thread or len(lines) % thread != 0:
        print("Error: File log contains incomplete or invalid information.")
        return num_epochs_completed, None

    # Get the information of the last epoch
    last_epoch_info = lines[-5:]
    last_epoch_train_loss = float(last_epoch_info[0].split(":")[1].strip())
    last_epoch_train_accuracy = float(last_epoch_info[1].split(":")[1].strip()[:-1])
    return num_epochs_completed, (last_epoch_train_loss, last_epoch_train_accuracy)
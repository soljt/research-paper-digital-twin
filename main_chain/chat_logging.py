import logging
import os
import sys

# Check if a session number file exists and read the session number, else initialize it
def read_and_update_session_number(log_dir):
    SESSION_FILE = f"{log_dir}/session_number.txt"
    # Try to read the current session number from the file
    try:
        with open(SESSION_FILE, "r") as f:
            session_number = int(f.read().strip())
    except FileNotFoundError:
        # If the file doesn't exist, start with session 000
        session_number = 0
    
    # Increment the session number for the next session
    new_session_number = session_number + 1
    
    # Write the updated session number back to the file
    with open(SESSION_FILE, "w") as f:
        f.write(str(new_session_number))
    
    return session_number  # Return the current session number before incrementing

def setup_logger(log_dir="./chat_logs", prefix="digital_dirk_output_", extension=".log"):

    session_number = read_and_update_session_number()

    # Generate the filename for the current session
    filename = f"{prefix}{session_number:03}{extension}"

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler to write to a log file
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(logging.INFO)

    # Stream handler to output to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

class ConsoleLogger:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout  # Save original stdout
        self.stdin = sys.stdin    # Save original stdin

    def write(self, text):
        self.stdout.write(text)  # Print to console
        self.file.write(text)    # Write to file

    def readline(self):
        """Read input from the console and log it to the file."""
        user_input = self.stdin.readline()
        self.file.write(f"You: {user_input}")  # Log user input
        return user_input

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
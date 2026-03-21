import sys

class Tee:
    """Redirects stdout to both console and a file."""
    
    def __init__(self, log_path):
        """
        Initialize Tee with file path.
        
        Parameters
        ----------
        log_path : str
            Path to the log file to write to
        """
        self.log_path = log_path
        self.log_file = open(log_path, 'w', buffering=1)  # Line buffered
        self.console = sys.stdout
        
        # Replace stdout with this Tee instance
        sys.stdout = self
    
    def write(self, data):
        """Write data to both console and log file."""
        try:
            self.console.write(data)
            self.console.flush()
        except Exception as e:
            pass  # Ignore console write errors
        
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except Exception as e:
            pass  # Ignore file write errors
    
    def flush(self):
        """Flush both streams."""
        try:
            self.console.flush()
        except:
            pass
        try:
            self.log_file.flush()
        except:
            pass
    
    def close(self):
        """Close the log file and restore stdout."""
        try:
            self.flush()
            self.log_file.close()
        except Exception as e:
            print(f"[Warning] Error closing log file: {e}", file=self.console)
        
        # Restore stdout
        sys.stdout = self.console
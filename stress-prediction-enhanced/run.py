import subprocess
import sys

# Step 1: Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Step 2: Run the Flask app
subprocess.run([sys.executable, "app.py"])

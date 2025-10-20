import os
import subprocess

folder = "code"

for filename in os.listdir(folder)[5:]:
    if filename.endswith(".py"):
        filepath = os.path.join(folder, filename)
        print(f"🔹 Running {filename}...")
        subprocess.run(["python", filepath], check=True)

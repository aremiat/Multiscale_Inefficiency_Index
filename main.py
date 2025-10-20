import os
import subprocess

folder = os.path.dirname(__file__) + "/code"

for filename in os.listdir(folder):
    if filename.endswith(".py"):
        filepath = os.path.join(folder, filename)
        print(f"🔹 Running {filename}...")
        subprocess.run(["python", filepath], check=True)

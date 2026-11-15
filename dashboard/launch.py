import subprocess
import time
import webview

process = subprocess.Popen(["streamlit", "run", "app.py", "--server.headless", "true"])
time.sleep(2)

webview.create_window("My Dashboard", "http://localhost:8501", width=1200, height=800)
webview.start()

process.terminate()

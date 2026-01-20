"""
Entry point for Marine Mammal Proteome Explorer - Streamlit Version
Run with: streamlit run streamlit_app.py
Or: python app.py
"""

import subprocess
import sys

if __name__ == '__main__':
    # Run streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "streamlit_app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ])

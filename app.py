"""
Entry point for Marine Mammal Proteome Explorer - Dash Version
Run with: python app.py
Or: gunicorn proteome_explorer_ux2:server
"""

# Import the Dash app
from proteome_explorer_ux2 import app, server

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)

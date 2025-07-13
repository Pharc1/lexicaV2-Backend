
from flask import Flask
from flask_cors import CORS
from main import main
from documents import documents
import os
from dotenv import load_dotenv

from werkzeug.middleware.proxy_fix import ProxyFix

# Charger les variables d'environnement
load_dotenv()

def create_app():
    """Créer et configurer l'application Flask."""
    app = Flask(__name__)
    
    # Configuration CORS pour permettre les requêtes depuis le frontend React
    CORS(app, origins=["https://lexica.pharci.fr"])
    
    # Enregistrement des blueprints
    app.register_blueprint(main, url_prefix='/api')
    app.register_blueprint(documents, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

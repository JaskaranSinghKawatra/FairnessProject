from flask import Flask
from config import Config
from .celery_utils import make_celery


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['UPLOAD_FOLDER'] = '/main/uploads'    
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379'
    
    return app

app = create_app()
celery = make_celery(app)

from .main import main as main_blueprint
app.register_blueprint(main_blueprint)
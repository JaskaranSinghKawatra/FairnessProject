from flask import Flask
from config import Config
from .celery_utils import make_celery
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
# from flask_socketio import SocketIO
from flask_cors import CORS

db = SQLAlchemy()
# socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(Config)
    app.config['UPLOAD_FOLDER'] = '/main/uploads'    
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379'
    app.config['CELERY_INCLUDE'] = ['app.main.tasks']
    db.init_app(app)
    # socketio.init_app(app, cors_allowed_origins="*")

    migrate = Migrate(app, db)
    
    return app

app = create_app()
celery = make_celery(app)

from .main import main as main_blueprint
app.register_blueprint(main_blueprint)
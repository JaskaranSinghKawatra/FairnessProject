import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    CELERY_BROKER_URL = 'redis://localhost:6379/0' 
    CELERY_RESULT_BACKEND = 'redis://localhost:6379/1'
Setup Redis
The host system needs to have Docker installed. To setup a Redis container the command below
needs to be run.

docker run -d --name some-redis -p 6379:6379 redis redis-server --timeout 3600

Setup Database
To initialize the SQLAlchemy database, the following commands are required to be executed in
the terminal:

flask db init
flask db migrate -m "Initial Commit"
flask db upgrade

Start Celery Workers
To begin the Celery workers to train the models in the background, the following command
needs to be executed in the terminal before the web application is started:

celery -A celery_worker.celery worker -l info -P gevent -c 4

Start Script
To begin the web application, the following command needs to be executed:

python manage.py

These commands need to be executed in the flask_project directory.

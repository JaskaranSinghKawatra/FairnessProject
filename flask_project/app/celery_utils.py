from celery import Celery
from gevent import Timeout
from app.redis_utils import Redis


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    celery.conf['BROKER_TRANSPORT_OPTIONS'] = {'visibility_timeout': 3600, 'socket_timeout': 3600} # 1 hour
    celery.conf['RESULT_BACKEND_TRANSPORT_OPTIONS'] = {'socket_timeout': 3600}

    class ContextTask(celery.Task):
        timeout = 3600

        def __call__(self, *args, **kwargs):
            with app.app_context():
                with Timeout(self.timeout, TimeoutError(f"Task {self.name} timed out")):
                    return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

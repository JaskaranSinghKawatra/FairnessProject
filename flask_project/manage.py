from app import app
from app import celery

if __name__ == '__main__':
    app.run(debug=True)
    print("Celery backend:", celery.backend)

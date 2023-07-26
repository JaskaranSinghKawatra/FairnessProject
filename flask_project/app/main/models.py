from app import db
from sqlalchemy.orm import relationship

class ModelResults(db.Model):

    __tablename__ = 'model_results'

    id = db.Column(db.String, primary_key=True)
    model_class = db.Column(db.String)
    fairness_notion = db.Column(db.String)
    learning_rate = db.Column(db.Float)
    lambda_fairness = db.Column(db.Float)
    batch_size = db.Column(db.Integer)
    num_epochs = db.Column(db.Integer)
    loss_values = db.Column(db.PickleType)
    accuracy_values = db.Column(db.PickleType)
    model_accuracy = db.Column(db.Float)
    auc_score = db.Column(db.Float)
    fairness_score = db.Column(db.Float) # Will be evaluated based on test set: For demographic parity, this is the maximum difference in prediction averages across the groups defined by the protected attributes

    fairness_metrics = relationship('FairnessMetrics', back_populates='model_results')


class FairnessMetrics(db.Model):

    __tablename__ = 'fairness_metrics'
    id = db.Column(db.String, primary_key=True)
    model_results_id = db.Column(db.String, db.ForeignKey('model_results.id'))
    fairness_notion = db.Column(db.String)
    group = db.Column(db.String)
    epoch = db.Column(db.Integer)
    metrics = db.Column(db.JSON)

    model_results = relationship('ModelResults', back_populates='fairness_metrics')
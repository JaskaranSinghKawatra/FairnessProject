"""Fresh

Revision ID: 7114b805146e
Revises: 6b755c02fc17
Create Date: 2023-07-27 13:36:28.661041

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7114b805146e'
down_revision = '6b755c02fc17'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('model_results',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('model_class', sa.String(), nullable=True),
    sa.Column('fairness_notion', sa.String(), nullable=True),
    sa.Column('learning_rate', sa.Float(), nullable=True),
    sa.Column('lambda_fairness', sa.Float(), nullable=True),
    sa.Column('batch_size', sa.Integer(), nullable=True),
    sa.Column('num_epochs', sa.Integer(), nullable=True),
    sa.Column('loss_values', sa.PickleType(), nullable=True),
    sa.Column('accuracy_values', sa.PickleType(), nullable=True),
    sa.Column('model_accuracy', sa.Float(), nullable=True),
    sa.Column('auc_score', sa.Float(), nullable=True),
    sa.Column('fairness_score', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('fairness_metrics',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('model_results_id', sa.String(), nullable=True),
    sa.Column('fairness_notion', sa.String(), nullable=True),
    sa.Column('group', sa.String(), nullable=True),
    sa.Column('epoch', sa.Integer(), nullable=True),
    sa.Column('metrics', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['model_results_id'], ['model_results.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('fairness_metrics')
    op.drop_table('model_results')
    # ### end Alembic commands ###

from flask import Blueprint, request, jsonify
from google.cloud import datastore

client = datastore.Client()
bp = Blueprint('crypto', __name__, url_prefix='/crypto')


@bp.route('/training_data', methods=['GET'])
def get_all_training_data():
    pass


@bp.route('/training_data/<data_name', methods=['GET', 'POST', 'DELETE'])
def training_data(data_name: str):
    if request.method == 'GET':
        return get_training_data(data_name)
    elif request.method == 'POST':
        return create_training_data(data_name)
    elif request.method == 'DELETE':
        return delete_training_data(data_name)


def get_training_data(data_name: str):
    pass


def create_training_data(data_name: str):
    pass


def delete_training_data(data_name: str):
    pass

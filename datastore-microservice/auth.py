from flask import Blueprint, request, jsonify
from google.cloud import datastore

client = datastore.Client()
bp = Blueprint('auth', __name__, url_prefix='/auth')


@bp.route("/", methods=['GET'])
def get_auth():
    return '', 501
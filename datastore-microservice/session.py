from flask import Blueprint, request, jsonify
from google.cloud import datastore
import datetime

client = datastore.Client()
bp = Blueprint('session', __name__, url_prefix='/session')


@bp.route('/', methods=['GET', 'POST'])
def sessions():
    if request.method == 'GET':
        return get_all_sessions()
    elif request.method == 'POST':
        return create_session(request)


def get_all_sessions():
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return 'Invalid value for limit. Limit argument must be an integer.', 400

    query = client.query(kind='Session')
    total = len(list(query.fetch()))
    session_iter = query.fetch(limit=limit, offset=offset)
    pages = session_iter.pages
    results = list(next(pages))
    if session_iter.next_page_token:
        new_offset = offset + limit
        next_url = f'{request.base_url}?limit={limit}&offset={new_offset}'
    else:
        next_url = None

    sessions = []
    for result in results:
        session = dict(result)
        session['session_id'] = result.id
        session['self'] = f'{request.base_url}{result.id}/transactions'
        sessions.append(session)

    return jsonify({
        "total": total,
        "sessions": sessions,
        "next": next_url if next_url is not None else ''
    })


def create_session(request):
    json = request.get_json()
    try:
        session_name = json['session_name']
        session_type = json['session_type']
        starting_balance = json['starting_balance']
        starting_coins = json['starting_coins']
        crypto_type = json['crypto_type']
    except KeyError:
        return jsonify({"error": "Missing required field(s)."})

    try:
        session_start = json['session_start']
    except KeyError:
        session_start = datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')

    


@bp.route('/<session_id>', methods=['GET', 'PATCH', 'DELETE'])
def specific_session(session_id: str):
    key = client.key('Session', int(session_id))
    session = client.get(key)

    if not session:
        return jsonify({"error": "No session with that session_id was found."}), 404

    if request.method == 'GET':
        return get_session(request, session)
    elif request.method == 'PATCH':
        return edit_session(request, session)
    elif request.method == 'DELETE':
        return delete_session(key, session)
    else:
        return '', 400


def get_session(request, result):
    session = dict(result)
    session['id'] = result.id
    session['transaction_url'] = f'{request.base_url}transactions'
    session['self'] = f'{request.base_url}'

    query = client.query(kind='Transaction')
    query.add_filter('session_id', '=', session.id)
    session['transaction_total'] = len(list(query.fetch))

    return jsonify(session), 200


def edit_session(session):
    return jsonify({"error": "Method not implemented."}), 501


def delete_session(key, session):
    pass


@bp.route('/<session_id>/transactions', methods=['GET', 'POST'])
def session_transactions(session_id: str):
    if request.method == 'POST':
        json = request.get_json()
        if json['type'] == 'BUY':
            return buy_crypto(session_id)
        elif json['type'] == 'SELL':
            return sell_crypto(session_id)
        else:
            return '', 400
    elif request.method == 'GET':
        get_transactions(session_id)
    else:
        return 'Not found: invalid request method.', 404


def get_transactions(session_id: str):
    query = client.query(kind='Trade')
    query.add_filter('session_id', '=', session_id)

    results = query.fetch()
    total = len(list(results))
    self = f'{request.base_url}'
    transactions = []
    for result in results:
        transaction = dict(result)
        transaction['transaction_id'] = result.id
        transactions.append(transaction)

    return jsonify({
        "self": self,
        "total_transactions": total,
        "transactions": transactions
    }), 200


@bp.route('/<session_id>/transactions/<transaction_id>', methods=['GET', 'DELETE'])
def transactions(session_id: str, transaction_id: str):
    key = client.key('Transaction', int(transaction_id))
    transaction = client.get(key)
    #query = client.query(kind='Transaction')
    #query.add_filter('session_id', '=', session_id)
    #query.add_filter('transaction_id', '=', transaction_id)
    #transaction = query.fetch()

    if not transaction or transaction['session_id'] != session_id:
        return jsonify({"error": "Transaction not found with that transaction id and/or session id."}), 404

    if request.method == 'GET':
        return get_transactions(transaction)
    elif request.method == 'DELETE':
        return delete_transaction(key)
    else:
        return '', 400


def get_transaction(result):
    transaction = dict(result)
    transaction['id'] = result.id
    transaction['self'] = f'{request.base_url}'

    return jsonify(transaction), 200


def delete_transaction(key):
    client.delete(key)
    return '', 204


def buy_crypto(session_id: str):
    pass


def sell_crypto(session_id: str):
    pass

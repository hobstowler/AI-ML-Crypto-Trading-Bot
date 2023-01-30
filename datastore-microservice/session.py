from flask import Blueprint, request, jsonify
from google.cloud import datastore
from datetime import datetime, timezone

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
    limit = 100 if limit > 100 else limit

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
        session['id'] = result.id
        session['self'] = f'{request.base_url}{result.id}/transactions'
        sessions.append(session)

    return jsonify({
        "total": total,
        "sessions": sessions,
        "next": next_url if next_url is not None else ''
    }), 200


def create_session(request):
    json = request.get_json()
    try:
        session_name = json['session_name']
        session_type = json['type']
        starting_balance = json['starting_balance']
        starting_coins = json['starting_coins']
        crypto_type = json['crypto_type']
    except KeyError:
        return jsonify({"error": "Missing required field(s)."}), 400

    try:
        session_start = json['session_start']
    except KeyError:
        session_start = datetime.utcnow().strftime('%m/%d/%Y, %H:%M:%S')

    session = datastore.Entity(client.key('Session'))
    session.update({
        'session_name': session_name,
        'type': session_type,
        'model_name': json['model_name'] if 'model_name' in json else None,
        'session_start': session_start,
        'session_end': json['session_end'] if 'session_end' in json else None,
        'starting_balance': starting_balance,
        'ending_balance': json['ending_balance'] if 'ending_balance' in json else None,
        'starting_coins': starting_coins,
        'ending_coins': json['ending_coins'] if 'ending_coins' in json else None,
        'coins_sold': json['coins_sold'] if 'coins_sold' in json else None,
        'coins_bought': json['coins_bought'] if 'coins_bought' in json else None,
        'number_of_trades': json['number_of_trades'] if 'number_of_trades' in json else None,
        'crypto_type': crypto_type,
    })

    client.put(session)
    res = dict(session)
    res['id'] = session.id
    res['self'] = f'{request.base_url}{session.id}'
    return jsonify(res), 201


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
    session['transaction_url'] = f'{request.base_url}/transactions'
    session['self'] = f'{request.base_url}'

    query = client.query(kind='Transaction')
    query.add_filter('id', '=', result.id)
    session['transaction_total'] = len(list(query.fetch()))

    return jsonify(session), 200


def edit_session(request, session):
    json = request.get_json()

    for key, value in json.items():
        session.update({key: value})
    """session.update({
        'session_name': json['session_name'] if 'session_name' in json else None,
        'type': json['type'] if 'type' in json else None,
        'model_name': json['model_name'] if 'model_name' in json else None,
        'session_start': json['session_start'] if 'session_start' in json else None,
        'session_end': json['session_end'] if 'session_end' in json else None,
        'starting_balance': json['starting_balance'] if 'starting_balance' in json else None,
        'ending_balance': json['ending_balance'] if 'ending_balance' in json else None,
        'starting_coins': json['starting_coins'] if 'starting_coins' in json else None,
        'ending_coins': json['ending_coins'] if 'ending_coins' in json else None,
        'coins_sold': json['coins_sold'] if 'coins_sold' in json else None,
        'coins_bought': json['coins_bought'] if 'coins_bought' in json else None,
        'number_of_trades': json['number_of_trades'] if 'number_of_trades' in json else None,
        'crypto_type': json['crypto_type'] if 'crypto_type' in json else None,
    })"""
    client.put(session)

    res = dict(session)
    res['id'] = session.id
    res['self'] = f'{request.base_url}{session.id}'
    return jsonify(res), 200


def delete_session(key, session):
    query = client.query(kind='Transaction')
    query.add_filter('session_id', '=', session.id)
    transactions = query.fetch()
    for t in transactions:
        client.delete(client.key('Transaction', int(t.id)))

    client.delete(key)
    return '', 204


@bp.route('/<session_id>/transactions', methods=['GET', 'POST'])
def session_transactions(session_id: str):
    if request.method == 'POST':
        return create_transaction(request)
    elif request.method == 'GET':
        return get_transactions(session_id)
    else:
        return 'Not found: invalid request method.', 404


def create_transaction(request):
    json = request.json()
    try:
        session_id = json['session_id']
        transaction_type = json['type']
        crypto_type = json['crypto_type']
        amount = json['amount']
        value = json['value']
    except KeyError:
        return jsonify({"error": "missing required fields"}), 404

    try:
        trans_dt_tm = json['transaction_date_time']
    except KeyError:
        trans_dt_tm = datetime.utcnow().strftime('%m/%d/%Y, %H:%M:%S')

    transaction = datastore.Entity(client.key('Transaction'))
    transaction.update({
        'session_id': session_id,
        'type': transaction_type,
        'crypto_type': crypto_type,
        'amount': amount,
        'value': value,
        'transaction_date_time': trans_dt_tm
    })
    client.put(transaction)
    res = dict(transaction)
    res['id'] = transaction.id
    res['self'] = f'{request.base_url}{transaction.id}'
    return jsonify(res), 201


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



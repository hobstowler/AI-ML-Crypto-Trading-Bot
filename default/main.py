from flask import Flask, request, jsonify
import requests

app = Flask(__name__, instance_relative_config=True)

URL = 'https://datastore-micro-dot-ai-ml-bitcoin-bot.uw.r.appspot.com/'


@app.route('/', methods=['GET'])
def index():
    return 'Refer to documentation at https://github.com/hobstowler/AI-ML-Bitcoin-Trading-Bot to begin. <br />' \
           'App reporting management: '


@app.route('/sessions', methods=['GET'])
def get_sessions():
    resp = requests.get(f'{URL}session')
    print(f'{URL}sessions', resp)
    if resp.status_code == 200:
        data = {}
        sessions = resp.json()['sessions']
        for session in sessions:
            model_name = session['model_name'] if session['model_name'] else '<no name>'
            if model_name in data.keys():
                data[model_name].append(session)
            else:
                data[model_name] = [session]
        return jsonify(data), 200
    else:
        return 'Error', resp.status_code

    return '', 500


@app.route('/session/<id>', methods=['GET'])
def get_session_by_id(id):
    resp = requests.get(f'{URL}session/{id}')
    if resp.status_code == 200:
        return resp.json()
    else:
        return '', resp.status_code


@app.route('/transactions', methods=['GET'])
def get_transactions():
    session_id = request.args.get("session_id", 'undefined')
    if session_id == 'undefined':
        return 'No session with provided ID found.', 404

    next_url = f'{URL}/session/{session_id}/transaction'
    results = []
    while next_url:
        resp = requests.get(next_url)
        if resp.status_code == 200:
            json = resp.json()
            next_url = json['next']
            transactions = json['transactions']
            for transaction in transactions:
                results.append(transaction)
        else:
            raise Exception(f'response from server: {resp.status_code}')
    # results.sort(key=lambda x: x['step'])

    sorted_results = {}
    for result in results:
        for key, val in result.items():
            if key not in ['step', 'id', 'session_id', 'type']:
                if key not in sorted_results.keys():
                    sorted_results[key] = [val]
                else:
                    sorted_results[key].append(val)

    return jsonify(sorted_results), 200


@app.route('/raw_transactions', methods=['GET'])
def get_raw_transactions():
    session_id = request.args.get("session_id", 'undefined')
    if session_id == 'undefined':
        return 'No session with provided ID found.', 404

    next_url = f'{URL}/session/{session_id}/transaction'
    results = []
    while next_url:
        resp = requests.get(next_url)
        if resp.status_code == 200:
            json = resp.json()
            next_url = json['next']
            transactions = json['transactions']
            for transaction in transactions:
                results.append(transaction)
        else:
            raise Exception(f'response from server: {resp.status_code}')
    # results.sort(key=lambda x: x['step'])

    return jsonify(results), 200



if __name__ == '__main__':
    app.run(port=4000)

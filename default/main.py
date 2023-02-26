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


@app.route('/transactions', methods=['GET'])
def get_transactions():
    session_id = request.args.get("session_id")
    if session_id == 'undefined':
        return 'No session with provided ID found.', 404


if __name__ == '__main__':
    app.run(port=4000)

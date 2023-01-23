from flask import Flask, request, jsonify
import crypto

app = Flask(__name__, instance_relative_config=True)
app.register_blueprint(crypto.bp)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'error':
                    'Refer to documentation at '
                    'https://github.com/hobstowler/AI-ML-Bitcoin-Trading-Bot to begin'}), 404

if __name__ == '__main__':
    app.run(port=2522)
from flask import Flask, request, jsonify, render_template
import inference

app = Flask(__name__, instance_relative_config=True)
app.register_blueprint(inference.bp)

print("Finished blueprint in main")

@app.route('/')
def root():

    return render_template(
        'index.html')

# @app.route('/', methods=['GET'])
# def index():
#     return 'Refer to documentation at ' \
#            'https://github.com/hobstowler/AI-ML-Bitcoin-Trading-Bot to begin', 404


if __name__ == '__main__':
    app.run(port=2600)

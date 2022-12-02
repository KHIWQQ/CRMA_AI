from flask import Flask, jsonify ,request

app = Flask(__name__)


@app.route('/keylogger', methods=['POST'])
def start():
    body = request.get_json()
    print(body)
    return 200


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from web2 import out


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/process-url', methods=['POST'])
def process_url():
    data = request.get_json()
    url = data.get('url', '')
    response_message=str(out(url))
    # response_message = f'Processed URL: {url}'
    
    return jsonify({"message": response_message})

if __name__ == '__main__':
    app.run(port=5000)

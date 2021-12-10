from flask import Flask, request, send_file, json
from flask_cors import CORS
from Inference import extract_from_text

app = Flask(__name__)
CORS(app)

@app.route('/extract', methods=['POST'])
def extract():

    res = {
        "code": 400,
        "data": {},
        "msg": "failure"
    }
    try:
        the_request = request.json
        the_request = dict(the_request)
        name = the_request.get('name')
        scope = the_request.get('scope')
        text_to_extract = name + '。' + scope
        text_to_extract = text_to_extract.replace(' ','')
        if len(text_to_extract) > 512:
            text_to_extract = text_to_extract[:512]
        result = extract_from_text(text_to_extract)

    except Exception as e:
        result = [str(e)]

    # return res
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8286)

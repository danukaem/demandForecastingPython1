from flask import Flask, jsonify, json, request

app = Flask(__name__)

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]


@app.route("/")
def hello():
    # return "Hello world"
    return jsonify({"about": "Hello world!"})


@app.route('/companies', methods=['GET'])
def get_companies():
    return json.dumps(companies)


@app.route('/html', methods=['GET'])
def get_html_button():
    query_parameters = request.args
    name= query_parameters.get('name')
    # return "<div><button>"+name+"</button></div>"
    return name


if __name__ == '__main__':
    app.run(debug=True)

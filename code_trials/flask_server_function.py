from flask import Flask, request, jsonify

app = Flask(__name__)

# Example: run a function that takes arguments
def my_function(x, y):
    return x + y

@app.route('/compute', methods=['GET'])
def compute():
    # Get query parameters (?x=...&y=...)
    try:
        x = float(request.args.get("x"))
        y = float(request.args.get("y"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing parameters"}), 400

    result = my_function(x, y)
    return jsonify({
        "x": x,
        "y": y,
        "result": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

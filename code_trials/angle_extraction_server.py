from flask import Flask, request, jsonify
from angle_extraction import Analyze_Image_Simple


app = Flask(__name__)

def main(filename):
    
    return Analyze_Image_Simple(filename)

@app.route('/compute', methods=['GET'])
def compute():
    # Get query parameters (?filename=...)
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Missing 'filename' parameter"}), 400

    try:
        angle = Analyze_Image_Simple(filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "filename": filename,
        "angle": angle
    })

# --- Ping route (health check) ---
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "Server is running"}), 200



@app.route('/shutdown', methods=['GET','POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return "Server shutting down..."

if __name__ == "__main__":
    

    app.run(host="0.0.0.0", port=5000, debug=True)

    

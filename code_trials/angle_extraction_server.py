from flask import Flask, request, jsonify, send_file
from angle_extraction import Analyze_Image_Simple, Analyze_Image

import os
from datetime import datetime

app = Flask(__name__)

def main(filename):
    
    return Analyze_Image_Simple(filename)

def analyze_image_with_plot(filename, save_plot=True):
    """
    Analyze image and optionally save plot to file
    
    Parameters:
    filename (str): Path to the image file
    save_plot (bool): Whether to save the generated plots
    
    Returns:
    tuple: (angle, plot_filename) where plot_filename is None if save_plot is False
    """
    try:
        # Create plots directory if it doesn't exist
        plots_dir = "plots"
        if save_plot and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Get base filename for plot naming
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = None
        plot_level=0
        if save_plot:
            import io
            import matplotlib
            matplotlib.use('Agg')  # Use a non-interactive backend
            import matplotlib.pyplot as plt
            plot_filename = os.path.join(plots_dir, f"{base_name}_{timestamp}_analysis.png")
            plot_level=1
        # Call Analyze_Image with plot_level=3 to generate plots

        angle_info = Analyze_Image(filename, plot_level=plot_level, verbose_level=0)
        angle = angle_info[0] if angle_info else None
        
        if save_plot and plt.get_fignums():  # Check if any figures exist
            # Save the current figure
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
        
        return angle, plot_filename
        
    except Exception as e:
        plt.close('all')  # Make sure to close figures even on error
        raise e

@app.route('/compute', methods=['GET'])
def compute():
    # Get query parameters (?filename=...)
    filename = request.args.get("filename")
    save_plot = request.args.get("save_plot", "true").lower() == "true"
    
    if not filename:
        return jsonify({"error": "Missing 'filename' parameter"}), 400

    try:
    
        angle, plot_filename = analyze_image_with_plot(filename, save_plot=save_plot)
        
        response_data = {
            "filename": filename,
            "angle": round(angle, 2) if angle is not None else None
        }
        
        if save_plot and plot_filename:
            response_data["plot_saved"] = plot_filename
            response_data["plot_url"] = f"/plot/{os.path.basename(plot_filename)}"
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/plot/<filename>', methods=['GET'])
def get_plot(filename):
    """Serve saved plot files"""
    plot_path = os.path.join("plots", filename)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Plot file not found"}), 404

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

    

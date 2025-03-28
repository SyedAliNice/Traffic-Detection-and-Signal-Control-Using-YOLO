
# 🚦 Traffic Detection and Signal Control Using YOLO
![image alt]()

#   📌 Overview
This project utilizes YOLO (You Only Look Once) for traffic detection and signal control. The system detects vehicles in a selected region, calculates traffic density, and dynamically adjusts the traffic signal accordingly.

#   🛠 Features
✅ Real-Time Vehicle Detection using YOLO.

✅ Polygon Selection for ROI (Region of Interest) to track vehicles.

✅ Dynamic Traffic Signal Control based on vehicle count.

✅ Customizable Parameters for fine-tuning the detection.

✅ Video Input and Output Processing for real-world applications.

#   📂 Project Structure
📁 Traffic Detection Project

    │-- main.py                   # Main script for detection and control
    │-- Traffic Detection.ipynb    # Jupyter Notebook for testing
    │-- yolo11s.pt                 # Pre-trained YOLO model

#   🔧 Installation
1️⃣ Clone the repository:

    git clone https://github.com/your-repo/traffic-detection.git
2️⃣ Navigate to the project directory:

    cd traffic-detection
3️⃣ Install required dependencies:

    pip install -r requirements.txt
4️⃣ Ensure OpenCV and Ultralytics YOLO are installed:

    pip install opencv-python ultralytics numpy
#   🚀 Usage
🔹 Run the main script:

    python main.py
 🔹 Modify parameters in main.py for custom configurations.

🔹 Use Jupyter Notebook (Traffic Detection.ipynb) for testing and visualization.
#   ⚙ Configuration
    Adjust parameters in main.py:
-   traffic_density_threshold ➝ Number of vehicles required to trigger signal change.
-   signal_duration ➝ Time duration for signal to remain ON.
-   update_interval ➝ Frame intervals for vehicle counting. 
#   📸 Demo
👉  The script processes a video (5.mp4) and outputs output_video2.mp4 showing real-time vehicle detection and signal control.  




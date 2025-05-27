# Welcome to Yannick's Project Showcase 🚀

Hi — I’m Yannick.  
I build systems that bridge physics, machine learning, and software engineering to solve complex, real-world problems.

This repository highlights a collection of projects built in high-stakes environments like aerospace — tools designed to detect anomalies, forecast power behavior from telemetry, and visualize control systems in real time. While many of these originated in the space domain, the engineering patterns and problem-solving approaches behind them are broadly applicable: clean architecture, robust pipelines, and a strong emphasis on observability and correctness.

---

## 📂 Projects Overview

### 1. [Spacecraft Battery Voltage Prediction Model 🔋](./Battery_Voltage_Prediction/spacecraft_battery_prediction_research.md)  
**Description**: This project showcases the foundational ML research behind a production-grade spacecraft battery monitoring system. I developed a pure ML model using a hybrid CNN-LSTM architecture to predict spacecraft battery voltage directly from telemetry. A novel "fractional orbit" feature—derived from eclipse transitions—enabled the model to infer orbital phase and learn charge/discharge patterns without explicit physics.

**Highlights**:
- Designed and trained a hybrid CNN-LSTM network with >1.25M parameters  
- Engineered the "fractional orbit" feature to encode orbital phase (0 to 1)  
- Achieved R² > 0.99 on validation data using only telemetry-derived inputs  
- Proved ML could model spacecraft power behavior, paving the way for physics-integrated production systems now running live on-orbit

---

### 2. [Reaction Wheel Friction Detection CLI Tool ⚙️](./rw_anomaly_detection/rw_friction_readme.md)  
**Description**: A command-line tool for detecting spacecraft reaction wheel anomalies using Isolation Forests, PCA, and engineered features from on-orbit telemetry. The tool features advanced preprocessing (Savitzky-Golay filtering, normalization), mode-based fault injection, and a complete ML pipeline for anomaly detection. While the telemetry interface is production-specific, the architecture and methodology demonstrate a full-scale operational detection system from raw telemetry to visualization.

---

### 3. [Spacecraft Attitude Visualization Tool 🌌](./OrbitVizProjectRoot/spacecraft_viz_readme.md)  
**Description**: A real-time 3D visualization tool built with Qt and VTK to simulate spacecraft attitude dynamics. This tool allows for interactive rendering of spacecraft orientation, reference vectors (Sun, Nadir, Velocity), and live quaternion-based updates in an intuitive GUI. Designed for mission operations and control engineers, it includes a physics-backed simulation engine, Dockerized deployment, and supports telemetry playback for post-flight analysis.

---

### 4. [Inverted Pendulum: Two-Wheeled Balancing Robot 🤖](./Inverted_Pendulum)  
**Description**: This project focused on stabilizing an inherently unstable two-wheeled robot using a state-space model and Linear Quadratic Regulator (LQR). By modeling the system as an inverted pendulum, I implemented a controller that maintains balance using feedback loops and real-time state estimation through Kalman filtering.

---

### 5. [Engineering Papers 📄](./Papers)  
A collection of engineering experiments and analyses, each showcasing a blend of theoretical, experimental, and practical approaches. Topics include:
- **Particle Image Velocimetry (PIV)**: Visualizing and quantifying fluid flow fields.  
- **Unsteady Drag Corrections**: Investigating drag forces on falling bluff bodies.  
- **Lift and Drag Studies**: Aerodynamic exploration in wind tunnels.  
- **2D Heat Conduction**: Understanding steady-state heat transfer.  
- **Dye Flow Visualization**: Analyzing fluid flow patterns with dye injection.

---

### 6. [Sudoku Solver 🧩](./Sudoku_Solver)  
A fun yet educational project where I built a Sudoku solver using a backtracking algorithm, complete with graphical visualization through `pygame`. This project demonstrates algorithmic problem-solving and provides a real-time view of how the solver fills each cell.

---

## A Bit About Me 🧑‍💻

I'm a systems-minded engineer with a background in control systems, modeling, and applied machine learning. I focus on building tools that are both principled and practical — from real-time diagnostics to predictive models grounded in real data.

My interests lie in:
- Building intelligent infrastructure and tooling  
- Bridging physical systems and ML  
- Solving hard problems with minimal fuss and maximal clarity  

**Connect:**  
- [LinkedIn](https://www.linkedin.com/in/yannab) 🌐  
- Email: bondoyannick9@gmail.com 📧

---

Feel free to explore the projects below. Each one reflects a different facet of how I think, build, and debug real-world systems.

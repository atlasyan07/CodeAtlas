# Welcome to Yannick's Project Showcase 🚀

Hello, and thanks for stopping by! I'm Yannick—a driven aerospace engineer with a passion for applying engineering principles to real-world challenges. While this repository contains a subset of my work, it offers a window into my journey through modeling, control systems, data-driven algorithms, and problem-solving. Below, you'll find projects that demonstrate engineering fundamentals in action, from predictive models to algorithmic solutions.

---

## 📂 Projects Overview

### 1. [Spacecraft Battery Voltage Prediction Model 🔋](./Battery_Voltage_Prediction)  
**Description**: This project showcases the foundational ML research behind a production-grade spacecraft battery monitoring system. I developed a pure ML model using a hybrid CNN-LSTM architecture to predict spacecraft battery voltage directly from telemetry. A novel "fractional orbit" feature—derived from eclipse transitions—enabled the model to infer orbital phase and learn charge/discharge patterns without explicit physics.

**Highlights**:
- Designed and trained a hybrid CNN-LSTM network with >1.25M parameters
- Engineered the "fractional orbit" feature to encode orbital phase (0 to 1)
- Achieved R² > 0.99 on validation data using only telemetry-derived inputs
- Proved ML could model spacecraft power behavior, paving the way for physics-integrated production systems now running live on-orbit

### 2. [Inverted Pendulum: Two-Wheeled Balancing Robot 🤖](./Inverted_Pendulum)  
**Description**: This project focused on stabilizing an inherently unstable two-wheeled robot using a state-space model and Linear Quadratic Regulator (LQR). By modeling the system as an inverted pendulum, I implemented a controller that maintains balance using feedback loops and real-time state estimation through Kalman filtering.

### 3. [Engineering Papers 📄](./Papers)  
A collection of engineering experiments and analyses, each showcasing a blend of theoretical, experimental, and practical approaches. Topics include:
- **Particle Image Velocimetry (PIV)**: Visualizing and quantifying fluid flow fields.  
- **Unsteady Drag Corrections**: Investigating drag forces on falling bluff bodies.  
- **Lift and Drag Studies**: Aerodynamic exploration in wind tunnels.  
- **2D Heat Conduction**: Understanding steady-state heat transfer.  
- **Dye Flow Visualization**: Analyzing fluid flow patterns with dye injection.

### 4. [Sudoku Solver 🧩](./Sudoku_Solver)  
A fun yet educational project where I built a Sudoku solver using a backtracking algorithm, complete with graphical visualization through `pygame`. This project demonstrates algorithmic problem-solving and provides a real-time view of how the solver fills each cell.

### 5. [Spacecraft Attitude Visualization Tool 🌌](./OrbitVizProjectRoot)  
**Description**: A real-time 3D visualization tool built with Qt and VTK to simulate spacecraft attitude dynamics. This tool allows for interactive rendering of spacecraft orientation, reference vectors (Sun, Nadir, Velocity), and live quaternion-based updates in an intuitive GUI. Designed for mission operations and control engineers, it includes a physics-backed simulation engine, Dockerized deployment, and supports telemetry playback for post-flight analysis.

---

## A Bit About Me 🧑‍🚀

I'm a practical engineer who enjoys blending core engineering principles with innovative solutions. Whether it's predicting battery health for satellites or visualizing fluid flow patterns, I strive to tackle complex problems with creativity and rigor. While this repository represents some of my shared work, it reflects a mindset focused on continuous learning and impactful contributions.

**Connect**:
- [LinkedIn](https://www.linkedin.com/in/yannab) 🌐
- Email: bondoyannick9@gmail.com 📧

---

Explore, reach out, or share your thoughts—I always enjoy connecting over engineering, data, or anything in between.

# Biomech-Vision

**A modular Computer Vision framework for real-time biomechanical analysis and sports telemetry using MediaPipe and OpenCV.**

---

## 💡 Engineering Philosophy
**Biomech-Vision** was engineered to solve a specific problem in sports tech: **Tight coupling between perception and logic.** Most computer vision scripts hardcode pixel coordinates (e.g., `if point[12].y > 500`), making them brittle and impossible to scale across different sports or camera angles.

My approach introduces a **Biomechanics Abstraction Layer**. By treating the human body as a semantic object rather than a list of array indices, the system achieves:
1.  **Subject Invariance:** The system normalizes data so it works identically on a 6-foot athlete or a child, without code changes.
2.  **Domain Agnosticism:** The same geometry engine powers Cricket, Rehab, and Gym analysis.
3.  **Maintainability:** Logic is readable English (`body.is_elbow_extended`), not "Magic Numbers."

---

## 📐 Mathematical Foundation
The core of the analysis relies on a custom **Geometry Engine** (`geometry.py`) that performs 3D vector arithmetic on pose landmarks.

### 1. Vector Extraction
Instead of analyzing raw coordinate points (X, Y), the system converts body segments into vectors relative to the joint. For an arm, we define two vectors originating from the elbow:

```python
# Vector = Destination - Origin
Vector_Shoulder = (X_shoulder - X_elbow,  Y_shoulder - Y_elbow)
Vector_Wrist    = (X_wrist    - X_elbow,  Y_wrist    - Y_elbow)
```

### 2. Angle Calculation (The Dot Product)
To determine the angle $\theta$ between these limbs, we utilize the Dot Product formula, derived from the Law of Cosines:

$$\vec{A} \cdot \vec{B} = ||\vec{A}|| \cdot ||\vec{B}|| \cos(\theta)$$

Solving for $\theta$:

$$\theta = \arccos \left( \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \cdot ||\vec{B}||} \right)$$




**Implementation Detail:**
Standard `arccos` functions can fail if floating-point errors push the cosine value slightly beyond $[-1, 1]$. My implementation includes a **clamping mechanism** (`np.clip`) to handle these edge cases, preventing runtime math errors during high-velocity movements.

---

## 🧠 Algorithmic Strategy & Heuristics
A major challenge in biomechanics is **Signal-to-Noise Ratio**. Analyzing every frame (30-60 per second) produces noisy data. We need to analyze specific "Key Events" (e.g., Ball Impact).

**Why not use AI?**
Training a secondary Neural Network to classify "Impact Frames" introduces latency and requires a massive labeled dataset.

**The Solution: The "Wrist Valley" Heuristic**
I implemented a kinematic signal processor based on the vertical trajectory of the bat swing.

* **The Concept:** During a cricket drive, the hands accelerate downwards (Gravity + Force) and then immediately reverse direction upwards (Follow-through).
* **The "Valley":** If you plot the Y-coordinate of the wrist over time, it forms a "V" shape. The bottom of this V (the Local Minimum) represents the lowest point of the swing.
* **The Trigger:** The system tracks the vertical velocity ($dy/dt$). When the gradient flips from positive (down) to negative (up), the system identifies the inflection point as the **Moment of Impact**.



This provides **O(1) complexity** event detection with zero added inference latency, allowing the system to isolate the exact millisecond of contact without human intervention.

---

## 🏗 System Architecture

### 1. The Perception Layer (MediaPipe)
Google MediaPipe provides the raw 33-point skeletal topology. I chose this over OpenPose for its lightweight mobile compatibility, allowing this architecture to eventually run on edge devices (phones/tablets).

### 2. The Abstraction Layer (`BodyState`)
A semantic wrapper that sanitizes raw data.
* **Responsibility:** Hides the complexity of MediaPipe indices.
* **Feature:** Converts normalized coordinates (0.0-1.0) into metric-approximations for meaningful velocity calculations.

### 3. The Logic Layer (Plugins)
Analysis logic is injected as plugins.
* `CricketDriveAnalyzer`: Checks specific constraints (e.g., Lead Elbow Extension > 165°).
* **Extensibility:** New sports can be added by creating a subclass of `BaseAnalyzer`, without touching the core computer vision pipeline.

---

## 🚧 Development Roadmap

### Phase 1: Core Logic (Current)
- [x] **Geometry Engine:** Vector math implementation for 3D angle extraction.
- [x] **Body Abstraction:** Semantic wrapper for raw pose landmarks.
- [ ] **Heuristic Detection:** Implementing the "Wrist Valley" algorithm for impact detection.

### Phase 2: Distributed Systems (Architecture)
- [ ] **Queue System:** Integrating **Redis** to decouple video ingestion from processing (Producer-Consumer).
- [ ] **Containerization:** Dockerizing the worker for cloud deployment.
- [ ] **API:** Node.js entry point for client applications.

---

## 🔮 Future Scope
The goal is to evolve this from a script into a **Biomechanical Platform**. By exposing the `BodyState` and `Geometry` modules via an API, this system can serve as the backend for:
* Remote Physical Therapy monitoring.
* Automated Gym Spotting assistants.
* Professional Sports Coaching telemetry.

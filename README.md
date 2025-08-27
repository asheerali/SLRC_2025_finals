

# PixieBots – SLRC 2025 Finalists

Welcome to the official repository for the PixieBots’ robot, a top-five finalist in **SLRC 2025 (Sri Lanka’s Largest Robotic Competition)**, University Category. This repository showcases our engineering journey, robust software, and mechanical innovations designed to tackle a complex, multi-faceted challenge under a demanding 30-minute time limit.

---

## 🚀 Robot Overview

Our robot features a compact, robust mechanical design (**24 cm x 18 cm**) and a custom **4-Degrees of Freedom (DoF) robotic arm**. It’s engineered for precision, adaptability, and autonomous operation in dynamic, unpredictable environments.

### Key Mechanical Features
- **4DOF Robotic Arm:** Shoulder rotation, elbow flexion, wrist rotation, and gripper actuation powered by precision servos.
- **Integrated Storage:** Two onboard compartments for color-classified ping-pong balls.
- **Durable Chassis:** Designed for stability across ramps, muddy regions, and confined passages.

<img src="https://github.com/user-attachments/assets/5353fcd7-91c0-462a-a587-c2c1568617ba" width="500">

---

## 🤖 Autonomous Capabilities

Our robot’s intelligence is built on a robust, modular codebase, enabling it to interpret and react to its surroundings with high accuracy and reliability.

### Navigation & Sensing
- **Line Following & Junction Detection:** Tracks both solid and colored lines, recognizes junctions for grid-based navigation.
- **Dotted Line Following:** Advanced tracking for segmented paths.
- **Obstacle Detection:** Proactively avoids obstacles using real-time sensor data.
- **Ramp Navigation:** Detects inclines (e.g., 20-degree ramp) and dynamically adjusts speed for safe traversal.

### Sensing & Control Systems
- **Color-Based Classification:** Detects and sorts ping-pong balls by color using calibrated sensors.
- **Barcode Reading:** IR sensor-based barcode detection for environmental awareness.
- **Encoder PID Control:** Maintains precise straight-line movement using PID algorithms and high-resolution encoders.

### Manipulation Tasks
- **Precise Object Handling:** Picks and places balls and boxes with accuracy.
- **Automated Sorting:** Stores classified objects in dedicated compartments.

[![Watch the video](https://img.youtube.com/vi/zhswnPzu45Y/0.jpg)](https://youtu.be/zhswnPzu45Y)

---

## 🧠 Robust Code Architecture

Our software is designed for reliability, flexibility, and real-time performance.

### Reliability Features
- **Comprehensive Error Handling:** Detects hardware faults, sensor anomalies, and recovers gracefully from unexpected events.
- **Fail-Safe Routines:** Watchdog timers and safety interlocks ensure safe operation at all times.

### Modularity & Extensibility
- **Task-Based Modules:** Each competition objective is encapsulated for rapid adaptation before/during the event.
- **Subsystem Abstraction:** Independent modules for locomotion, manipulation, and sensing allow parallel development and debugging.

### Real-Time Decision Making
- **Event-Driven Architecture:** Asynchronous event loops respond instantly to environmental changes and judge commands.
- **Subsystem Communication:** Efficient message passing coordinates the arm, drive system, and sensors for multi-step tasks.

---

## 🦾 The 4DOF Robotic Arm – Technical Highlights

A centerpiece of our robot, the 4DOF arm, enabled success in manipulation and sorting tasks.

### Mechanical Design
- **Joints:** Shoulder rotation, elbow flexion, wrist rotation, gripper actuation.
- **Actuators:** High-precision servos for smooth, controlled movement.
- **Workspace:** Optimized kinematic chain for object pickup, placement, and interaction.

### Control System
- **Kinematics:** Forward and inverse kinematics for accurate end-effector positioning.
- **Feedback:** Encoders and current sensors for closed-loop force and position control.
- **Programmability:** Supports custom trajectories and variable-speed operations for delicate and fast tasks.

### Role in Competition
- **Task Execution:** Essential for picking, sorting, and placing objects under time pressure.
- **Contribution:** Key to completing color classification, ball collection, and storage tasks.

---

## 🏆 Competition Journey

The SLRC 2025 finals presented us with eight distinct objectives to be completed within a tight 30-minute window. Our robot excelled in:

1. **Ping-Pong Ball Collection:** Grabbing balls from cylindrical components in a grid.
2. **Color-Based Classification:** Sorting balls by color with high sensor precision.
3. **Muddy Region Navigation:** Maneuvering through tough terrain and obstacles.
4. **Ramp Ascent & Descent:** Traversing steep ramps with robust control.
5. **Confined Passage Navigation:** Precise steering through narrow, walled passages.

Our seamless transitions between environments and complex multi-step execution propelled us to the top five.

---

## 📂 Repository Contents

- Source code: Algorithms, sensor integration, and control systems
- Technical documentation: Arm schematics, system architecture, setup instructions
- Competition media: Photos, videos, and performance highlights

[![Watch on YouTube Shorts](https://img.youtube.com/vi/lcyCcpWFLiU/0.jpg)](https://youtube.com/shorts/lcyCcpWFLiU)

---

## 🎥 Grand Finale Video – SLRC 2025

Watch our robot's performance in the official grand finale broadcast:  
**Performance Period:** [5:56:00 to 6:12:00](https://m.youtube.com/watch?v=aplZ_r-cd3Y&t=21360s)  
Full broadcast: [SLRC 2025 Finals Live Broadcast](https://m.youtube.com/watch?v=aplZ_r-cd3Y)

---

## 📖 References

- [SLRC University Category 2025 Tasks (PDF)](https://ent.uom.lk/wp-content/uploads/2025/02/SLRC-University-Category.pdf)

---

We hope this repository provides valuable insights into our engineering process and inspires future robotic endeavors!  
**Questions or feedback?** Feel free to open an issue or pull request.

---
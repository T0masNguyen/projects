# TDoA Downlink Positioning System

## Project Overview

This project is focused on accurately determining the position of a tag within an area using multiple anchors as signal transmitters. The system leverages **Time Difference of Arrival (TDoA)** to estimate the tag's location, ensuring high precision through various algorithms and synchronization techniques.

### System Components

- **Anchors and Tag**:
  - **Anchors**: These are the fixed devices placed in the environment that transmit signals.
  - **Tag**: This device receives signals from the anchors, and the goal is to estimate its position in the environment.

- **Data Collection and Communication**:
  - A **router** within the office network helps facilitate communication between the anchors and the central processing unit (PC).
  - The PC continuously gathers data from both the anchors and the tag.
  - The system uses **Network Time Protocol (NTP)** to synchronize all devices' clocks, ensuring accurate timing measurements.
  - Data is transferred using **MQTT (Message Queuing Telemetry Transport)**, a lightweight messaging protocol that allows for efficient data communication.

### Position Estimation Algorithms

To calculate the position of the tag, the system uses different algorithms:

- **Least Squares Estimation (LSE)**:
  - A basic method that minimizes the squared differences between the observed and calculated time differences.
  - Provides an initial estimate of the tag’s position but may need further refinement for better accuracy.

- **Chan's Algorithm**:
  - A more advanced algorithm that improves the basic LSE by considering the geometric relationships between the anchors and the tag.
  - It's widely used as a benchmark in TDoA systems for its reliability and accuracy.

- **Taylor Series Expansion (Taylor)**:
  - An iterative method that improves the position estimate by linearizing the range equations and solving them step by step.
  - This method is particularly useful when higher accuracy is required, as it fine-tunes initial estimates provided by other methods.

### Clock Frequency Testing and Optimization

- **Clock Synchronization**:
  - To further enhance the system's accuracy, the clock frequencies of the anchors and the tag are tested and optimized. This helps to reduce timing errors, which can significantly affect the position estimation.

### Conclusion

This project demonstrates the effective use of a TDoA Downlink Positioning System for accurate tag location estimation. By combining several position estimation algorithms, efficient synchronization, and reliable data transfer, the system ensures that the results are precise and trustworthy. This project serves as a strong foundation for further improvements and potential real-world applications.

![Result](https://github.com/user-attachments/assets/ebb3be4d-ed3f-4a86-994f-a6d3abf858d7)

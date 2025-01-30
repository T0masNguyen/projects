# TDoA Downlink Positioning System

## Project Description

This project implements a Time Difference of Arrival (TDoA) Downlink Positioning System, designed to estimate the position of a tag in a given environment using multiple anchors as signal transmitters. The system is built to support high accuracy positioning by leveraging various algorithms and techniques to mitigate errors, improve synchronization, and ensure reliable data transfer.

### System Overview

- **Anchors and Tag**: 
  - Several anchors are deployed to serve as signal transmitters.
  - A single tag is used to receive these signals.
  - The anchors are strategically placed in the environment to cover the desired area for positioning.

- **Data Collection and Communication**: 
  - A router within the office network is utilized to facilitate communication between the anchors and the central processing unit (PC).
  - The PC continuously pulls data from the anchors and the tag.
  - **Network Time Protocol (NTP)** is employed to synchronize the clocks of all devices involved, ensuring that the timing measurements are as accurate as possible.
  - Data is transferred using **MQTT (Message Queuing Telemetry Transport)**, a lightweight messaging protocol ideal for efficient data communication in the system.

### Position Estimation Algorithms

To estimate the position of the tag, several algorithms were implemented and tested:

- **Least Squares Estimation (LSE)**: 
  - A basic approach that minimizes the sum of the squared differences between the observed and calculated time differences.
  - Provides an initial estimate of the tag's position but may require refinement for higher accuracy.

- **Chan's Algorithm**: 
  - A more advanced algorithm that improves upon the basic LSE by considering the geometric relationships between the anchors and the tag.
  - Often used as a benchmark for TDoA systems due to its reliability and accuracy.

- **Taylor Series Expansion (Taylor)**: 
  - This iterative method refines the position estimate by linearizing the range equations and solving them iteratively.
  - Particularly useful when higher accuracy is needed, as it can fine-tune the initial estimates provided by other methods.

### Clock Frequency Testing and Optimization

- **Clock Synchronization**: 
  - To further enhance the accuracy of the system, the clock frequencies of the anchors and tag were tested and optimized. This process helps in minimizing the timing errors that can significantly affect the accuracy of the position estimates.

### Conclusion

This project demonstrates a robust implementation of a TDoA Downlink Positioning System, capable of accurately determining the position of a tag within a defined area. The integration of various position estimation algorithms, combined with meticulous synchronization and data transfer mechanisms, ensures that the system provides reliable and accurate results. The work done here lays a solid foundation for future enhancements and applications in various real-world scenarios.

![result](https://github.com/user-attachments/assets/ebb3be4d-ed3f-4a86-994f-a6d3abf858d7)

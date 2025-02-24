# Improved AI-based Obstacle Avoidance for Autonomous Underwater Vehicles

[TODO]: This repository will contain the code implementation, the neural network itself, and some further information on how to set up and run the AI-based controller for Autonomous Underwater Vehicles (AUVs).

[TODO]: Some short description of this repository / link to published paper / aXiv version

## Setup (tested under Ubuntu 20.04)

### System requirements

- 2 GB available RAM memory
- Intel core i3 or better CPU
- 512 MB of hard drive space
- Ubuntu 20.04 operating system

### Install the [LSTS Neptus Command & Control Interface](https://github.com/LSTS/neptus)

1. Clone the public, open-source repository from GitHub:
```
git clone https://github.com/LSTS/neptus.git
```

2. Enter the cloned repository and run the `gradlew` build script:
```
cd neputs && ./gredlew
```

### Install the [LSTS DUNE Unified Navigation Environment](https://github.com/LSTS/dune)

1. Clone the public, open-source repository from GitHub:
```
git clone https://github.com/LSTS/dune.git
```

2. Enter the cloned repository and run the `cmake` build script:
```
cd dune && mkdir build && cd build && cmake .. && make -j4
```

### Run a simple simulated mission

1. Start Neptus:
```
cd neptus && ./neptus
```

2. Start dune:
```
cd dune && ./dune -c lauv-simulator-1 -p Simulation
```

3. Wait until the `lauv-simulator-1` simulated LAUV appears in Neptus.

4. In the top right dropdown, select `lauv-simulator-1`.

5. Upload the plan `plan1` using the blue button under the dropdown.

6. Run the simulated mission using the green button on the right of the blue one.

7. Following this LAUV should execute the planned simulated mission.

### [TODO]: Integrating the neural network controller

<!--
- ONNX file of the NN controller
- Training scripts
- Running simulated missions with the NN controller
- Share info on verification of the controller / properties / available tools
-->

# Acknowledgements

<strong>This underwater robot controller was developed during a collaboration with Oceanscan-MST, within the scope of H2020 [REMARO](https://remaro.eu/) project.</strong> 

More information about Oceanscan-MST can be found at this [link](https://www.oceanscan-mst.com/).

<a href="https://remaro.eu/">
    <img height="60" alt="REMARO Logo" src="https://remaro.eu/wp-content/uploads/2020/09/remaro1-right-1024.png">
</a>
<a href="https://www.oceanscan-mst.com/">
    <img height="60" alt="REMARO Logo" src="https://isola-project.eu/wp-content/uploads/2020/07/OceanScan.png">
</a>

This work is part of the Reliable AI for Marine Robotics (REMARO) Project. For more info, please visit: <a href="https://remaro.eu/">https://remaro.eu/

<br>

<a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en">
    <img align="left" height="60" alt="EU Flag" src="https://remaro.eu/wp-content/uploads/2020/09/flag_yellow_low.jpg">
</a>

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956200.



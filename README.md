# Traffic-state-reconstruction-using-Deep-CNN

In this study, we propose a statistical learning-based traffic speed estimation method that uses sparse vehicle trajectory information. Using a convolutional encoder-decoder based architecture, we show that a well trained neural network can learn spatio-temporal traffic speed dynamics from timespace diagrams. We demonstrate this for a homogeneous road section using simulated vehicle trajectories and then validate it using real-world data from NGSIM. Our results show that with probe vehicle penetration levels as low as 5%, the proposed estimation method can provide a sound reconstruction of macroscopic traffic speeds and reproduce realistic shockwave patterns, implying applicability in a variety of traffic conditions. We further discuss the model’s reconstruction mechanisms and confirm its ability to differentiate various traffic behaviors such as congested and free-flow traffic states, transition dynamics, and shockwave propagation.

The research paper can be found here: https://ieeexplore.ieee.org/document/9107074

### Some reconstruction results

<img src="Trained%20models/Reconstruction.PNG" width="600">

### Learned filters relevant for traffic flow reconstruction

<img src="Trained%20models/LearnedFilters.PNG" width="600">


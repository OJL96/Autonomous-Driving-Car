# _Programming an Autonomous Driving Car using Deep Learning and Object Detection_

### Project Description 
Using a Pi-Car with a mounted camera, a two-model framework has been created to navigate the car around mulitple tracks and responding correctly to objects along the way. The simulation has been kept simple with only two variables to be considered: the angle to which the car turns, and the speed. Images are fed into the model as the car navigates through the track (the number of images passed through depends on the inference time of model). A basic CNN model responsible for returning the speed and angle values if no objects are detected, while the other model uses the MobileNet V2 Architecture with SSD for detecting objects to which a different speed and angle value are returned depending on what objects have been detected plus other conditions. The scope of this project is to provide a robust framework that performs better than the original method. Please read the report for a more detailed analysis.

### Contents
> [Kaggle Challenege and Data](https://www.kaggle.com/c/machine-learning-in-science-2021)

> [Autonomous Driving Car Report](https://github.com/OJL96/MLP2_CW/files/6710157/MLiSP2.-.Report.pdf) (Detailed analysis + reference list found here)

> [Presentation of Methodology](https://web.microsoftstream.com/video/9c2bc0a1-8020-42a4-b12c-dda15e6eac50)

> [Model Testing](https://youtu.be/YwOy9E1MHm0)



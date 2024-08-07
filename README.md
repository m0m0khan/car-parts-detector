# Car parts detector

A deep learning model to detect the visibility of the hood and left backdoor of a car.

## Table of Contents

- [Car parts detector](#project-name)
  - [About](#about)
  - [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Inference](#inference)

## About

The project is a deep learning task in order to predict a score for the perspective of two car components. This score assumes values between 0.0 and 1.0. The value is 1.0 if the car component is photographed completely and from the front and 0.0 if the car component is not visible.


## Getting Started

Instructions on how to get a copy of the project running on your local machine for development and testing purposes.


### Prerequisites

Docker

### Installation

Step-by-step guide on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/m0m0khan/car-parts-detector.git

# Navigate to the app directory
cd car-parts-detector/api

# Build using Dockerfile
docker build -t car-parts-inference:latest .

# Run the docker image
docker run -p 5000:5000 car-parts-inference:latest
```

### Inference

Either of the ways can be used to test the model on an image of a car.

- The results can be inferred by using curl.

```bash
curl -X POST http://localhost:5000/predict -F "file=@/path/to/image.jpg"
```

There is a test image in the example folder for inferring the results. It can be used to predict the results as:

```bash
cd car-parts-detector/example
curl -X POST http://localhost:5000/predict -F "file=@test_1.jpg"

# Response
{"backdoor_probability":0.10644196718931198,"hood_probability":0.871918797492981}
```

- The results can also be inferred by using Postman.

  ![image](https://github.com/user-attachments/assets/7a3a37d3-acf6-44a7-a249-26e0a26afa50)

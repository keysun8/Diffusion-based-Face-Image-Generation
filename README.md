# Diffusion-based-Face-Image-Generation

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)

Link : ![Demo](https://huggingface.co/spaces/keysun89/this_person_does_not_exist_ldm)

<!-- Please add a screenshot or GIF of your project working here -->
![Demo/Screenshot](https://via.placeholder.com/800x400?text=Add+Project+Screenshot+Here)

## Overview
[Provide a brief overview of your project. Example: This is a simple image classification Flask app trained on the top of Keras API. The trained model takes an image as an input and classifies the class of image...]

## Motivation
[Explain why you built this project. Example: What could be a perfect way to utilize the weekend? I couldn't find any relevant research paper or dataset associated with it. And that led me to collect the images to train a deep learning model...]

## Technical Aspect
[Break down the technical implementation details of your project. Example:]
This project is divided into two parts:
1. Training a deep learning model using Keras. *(Not covered in this repo)*
2. Building and hosting a Flask web app. 
- A user can choose an image from a device or capture it using a pre-built camera.
- Used Amazon S3 Bucket to store the uploaded images.
- Used CSRF Token to protect against CSRF attacks.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

```bash
pip install -r requirements.txt

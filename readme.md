# Rain Prediction in Australia

## Overview
This repository contains code to predict  whether it will rain tomorrow in Australia based on today's weather data. 
It uses a trained Random Forest model and a Streamlit app.

## Project Structure
```
project/
│
├── data/               # Dataset file
├── images/             # Image for the Streamlet app
├── models/             # Trained model file
├── app.py              # Streamlit application script
├── requirements.txt    # Dependencies required for the project
└── train.ipynb         # Jupyter notebook for model training	
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/project-name.git
cd project-name
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model
You can train the model using the Jupyter notebook:
```
jupyter notebook train.ipynb
```

### Running the Application
To run the main application:
```
python app.py
```

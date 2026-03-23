
# Titanic DL Survival Predictor

This repository contains a Deep Learning model and a Streamlit web application for predicting the survival chance of Titanic passengers.

## Features

*   Interactive input widgets for passenger details (class, gender, family size, fare, embarkation port).
*   Utilizes a pre-trained Keras/TensorFlow Deep Learning model.
*   Includes saved preprocessing objects (StandardScaler, OneHotEncoder, LabelEncoder).
*   Real-time prediction and display of survival probability and a textual outcome.

## Setup and Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/titanic-dl-survival-predictor.git
    cd titanic-dl-survival-predictor
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n titanic_env python=3.11
    conda activate titanic_env
    ```
    *(If you used a path-based `venv`, specify the path: `conda activate ./venv`)*

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the app in your web browser.

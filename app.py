import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model # Assuming Keras is installed via tensorflow

# --- Configuration and Setup ---
# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Titanic Survival Predictor")

# Set random seed for reproducibility (if your model training also used seeds)
# This isn't strictly necessary for just prediction, but good practice if any randomness
# in preprocessing could affect results (e.g., some imputation methods).
# np.random.seed(42)
# tf.random.set_seed(42) # Only if you import tensorflow as tf directly

# --- Caching for Efficiency ---
# Load model and preprocessing objects only once when the app starts
@st.cache_resource # Use st.cache_resource for models and large objects
def load_all_assets():
    try:
        model = load_model('model.h5')
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
        with open('onehot_encoder.pkl', 'rb') as file:
            onehot_encoder = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, label_encoder, onehot_encoder, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Please ensure 'model.h5', 'label_encoder.pkl', 'onehot_encoder.pkl', and 'scaler.pkl' are in the same directory as this script.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()

model, label_encoder, onehot_encoder, scaler = load_all_assets()

# --- Helper Function for Prediction ---
def predict_survival(pclass, sex, sibsp, parch, fare, embarked, label_enc, onehot_enc, scaler_obj, model_obj):
    # Create DataFrame from user inputs
    user_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])

    # Apply label encoder for 'Sex'
    user_data['Sex'] = label_enc.transform(user_data[['Sex']])

    # Apply one-hot encoder for 'Embarked'
    embarked_encoded = onehot_enc.transform(user_data[['Embarked']])
    embarked_df = pd.DataFrame(embarked_encoded, columns=onehot_enc.get_feature_names_out())

    # Concatenate numerical and one-hot encoded features
    user_data = pd.concat([user_data.drop(columns=['Embarked']), embarked_df], axis=1)

    # Apply scaler for numerical features
    numerical_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
    user_data[numerical_cols] = scaler_obj.transform(user_data[numerical_cols])
    
    # Ensure all columns are in the correct order and count as the training data expects
    # This is crucial for consistent predictions. You might need to adjust this
    # if your training data had other columns or a different column order.
    # For robust deployment, you'd typically save the final column order from training.
    # Example placeholder for ordered columns:
    # expected_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    # user_data = user_data[expected_columns]

    # Make prediction
    prediction_array = model_obj.predict(user_data)
    probability = prediction_array[0][0] # Extract scalar probability

    return probability

# --- Function to interpret probability into text ---
def get_survival_text(probability):
    if probability > 0.5:
        return 'The Passenger will survive the Journey!', 'green'
    else:
        return 'The Passenger won\'t survive the Journey.', 'red'

# --- Streamlit UI Layout ---

st.title("🚢 Titanic Survival Chance Predictor")
st.markdown("""
    Enter the passenger's details below to predict their survival probability on the Titanic.
    Adjust the inputs using the sliders and dropdowns in the sidebar.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("Passenger Details")

pclass = st.sidebar.slider('Passenger Class (1st, 2nd, 3rd)', 1, 3, 1, help="1=1st, 2=2nd, 3=3rd class")
sex_input = st.sidebar.selectbox('Gender', ['male', 'female'], help="Male or Female")
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0, help="Number of siblings or spouses traveling with the passenger")
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0, help="Number of parents or children traveling with the passenger")
fare = st.sidebar.number_input('Fare Paid ($)', min_value=0.0, value=32.20, help="The fare paid for the ticket")
# Corrected typo 'Chebourg' to 'Cherbourg'
embarked_port = st.sidebar.selectbox('Port of Embarkation', ['Southampton', 'Cherbourg', 'Queenstown'], help="Where the passenger boarded the ship")

st.sidebar.markdown("---") # Separator in sidebar

# --- Main Content Area for Prediction Button and Results ---
st.markdown("## Get Your Prediction")
if st.button('Predict Survival Chance'):
    with st.spinner('Calculating survival chance...'):
        # Make the prediction
        survival_probability = predict_survival(
            pclass, sex_input, sibsp, parch, fare, embarked_port,
            label_encoder, onehot_encoder, scaler, model
        )
        
        # Get textual interpretation
        result_text, color = get_survival_text(survival_probability)

        st.subheader("Prediction Results:")
        
        # Display probability with st.metric for a nicer look
        st.metric(label="Survival Probability", value=f"{survival_probability*100:.2f}%")
        
        # Display textual result with st.success or st.error
        if color == 'green':
            st.success(f"**Outcome:** {result_text}")
        else:
            st.error(f"**Outcome:** {result_text}")

        st.info("Remember: This is a prediction based on a trained model and statistical patterns in the Titanic dataset. It does not guarantee an actual outcome.")

# --- Footer or Additional Information (using an expander) ---
st.markdown("---")
with st.expander("About This Predictor"):
    st.markdown("""
        This application uses a Deep Learning model (built with TensorFlow/Keras) trained on the historical Titanic dataset. 
        It takes various passenger attributes as input to predict the likelihood of survival.
        
        **Model Details:**
        *   Architecture: Simple Dense Neural Network
        *   Features used: Pclass, Gender, SibSp, Parch, Fare, Embarked
        *   Preprocessing: Label Encoding for Gender, One-Hot Encoding for Embarked, Standard Scaling for numerical features.
        
        *Disclaimer: This model is for educational purposes and demonstrates machine learning concepts.*
    """)

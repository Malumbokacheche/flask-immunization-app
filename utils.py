from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np

# Ordinal mappings for categorical variables
WEALTH_MAP = {'POOREST': 1, 'SECOND': 2, 'MIDDLE': 3, 'FOURTH': 4, 'RICHEST': 5}
EDUCATION_MAP = {'ECE': 1, 'PRIMARY': 2, 'LOWER SECONDARY': 3, 'UPPER SECONDARY': 4,
                 'HIGHER': 5, 'VOCATIONAL TRAINING': 6}
RADIO_MAP = {'NOT AT ALL': 1, 'LESS THAN ONCE A WEEK': 2, 'AT LEAST ONCE A WEEK': 3,
             'ALMOST EVERY DAY': 4, 'NO RESPONSE': 0}
MOBILE_MAP = {'NOT AT ALL': 1, 'LESS THAN ONCE A WEEK': 2, 'AT LEAST ONCE A WEEK': 3,
              'ALMOST EVERY DAY': 4, 'NO RESPONSE': 0}
PRENATAL_MAP = {'NO': 0, 'YES': 1}
MARITAL_MAP = {'NOT_IN_UNION': 0, 'LIVING_WITH_PARTNER': 1, 'MARRIED': 2}

# -------------------------------------------------------------------
# Map form field names to the exact feature names expected by the model
# -------------------------------------------------------------------
FIELD_TO_FEATURE = {
    'Age_of_child': 'Age_of_child',
    'Age_of_woman': 'Age_of_woman',
    'Wealth_index_quintile': 'Wealth_index_quintile',
    'Highest_level_of_school_attended': 'Highest_level_of_school_attended',
    'Frequency_of_listening_to_radio': 'Frequency_of_listening_to_the_radio',  # note long name
    'Mobile_phone_usage': 'Mobile_phone_usage_in_the_last_3_months',           # note long name
    'Received_prenatal_care': 'Received_prenatal_care',
    'Marital_Status': 'Marital_Status'  # if your model expects a different name, change here
}

def load_model(pkl_path='immunization_model.pkl'):
    """
    Load the full pipeline (preprocessor + classifier).
    Returns the pipeline and the feature names order.
    """
    data = joblib.load(pkl_path)
    # If the saved object is a dict (as in our training script)
    if isinstance(data, dict) and 'model' in data and 'preprocessor' in data:
        pipeline = Pipeline([
            ('preprocessor', data['preprocessor']),
            ('classifier', data['model'])
        ])
        feature_names = data['preprocessor'].feature_names_in_
        return pipeline, feature_names
    else:
        # If it's already a pipeline
        pipeline = data
        feature_names = pipeline.named_steps['preprocessor'].feature_names_in_
        return pipeline, feature_names

def compute_scores_and_interactions(form_data):
    """
    Convert user input (categorical + ages) into:
        - 6 score features
        - 8 interaction features (age * score)
        - binary flags: Has_Prenatal_Care, Is_Married
    Returns a dict with all derived features.
    """
    age_child = form_data['Age_of_child']
    age_woman = form_data['Age_of_woman']
    wealth = form_data['Wealth_index_quintile']
    education = form_data['Highest_level_of_school_attended']
    radio = form_data['Frequency_of_listening_to_radio']  # short name from form
    mobile = form_data['Mobile_phone_usage']              # short name from form
    prenatal = form_data['Received_prenatal_care']
    marital = form_data['Marital_Status']

    # Scores
    wealth_score = WEALTH_MAP[wealth]
    education_score = EDUCATION_MAP[education]
    radio_score = RADIO_MAP[radio]
    mobile_score = MOBILE_MAP[mobile]
    socioeconomic_score = (wealth_score + education_score) / 2.0
    if radio_score == 0 and mobile_score == 0:
        communication_score = 0
    elif radio_score == 0:
        communication_score = mobile_score
    elif mobile_score == 0:
        communication_score = radio_score
    else:
        communication_score = (mobile_score + radio_score) / 2.0

    prenatal_binary = PRENATAL_MAP[prenatal]
    married_binary = MARITAL_MAP[marital]  # 0,1,2; we treat 2 (married) as 1, others as 0
    is_married = 1 if married_binary == 2 else 0

    # Interactions
    age_wealth = age_child * wealth_score
    age_education = age_child * education_score
    age_socioeconomic = age_child * socioeconomic_score
    age_mobile = age_child * mobile_score
    age_radio = age_child * radio_score
    age_communication = age_child * communication_score
    age_prenatal = age_child * prenatal_binary
    age_married = age_child * is_married

    return {
        # Scores
        'Wealth_Score': wealth_score,
        'Education_Score': education_score,
        'Socioeconomic_Score': socioeconomic_score,
        'Mobile_Score': mobile_score,
        'Radio_Score': radio_score,
        'Communication_Score': communication_score,
        # Binary flags (these are expected as separate features)
        'Has_Prenatal_Care': prenatal_binary,
        'Is_Married': is_married,
        # Interactions
        'Age_Wealth_Interaction': age_wealth,
        'Age_Education_Interaction': age_education,
        'Age_Socioeconomic_Interaction': age_socioeconomic,
        'Age_Mobile_Interaction': age_mobile,
        'Age_Radio_Interaction': age_radio,
        'Age_Communication_Interaction': age_communication,
        'Age_Prenatal_Interaction': age_prenatal,
        'Age_Married_Interaction': age_married
    }

def prepare_input_for_model(form_data, feature_names):
    """
    Build a DataFrame with all features in the exact order expected by the model.
    Uses the FIELD_TO_FEATURE mapping to convert form field names to feature names.
    """
    # Get derived features (includes scores, binaries, interactions)
    derived = compute_scores_and_interactions(form_data)

    # Start with the raw categorical values using the correct feature names
    input_dict = {}
    for form_field, feature_name in FIELD_TO_FEATURE.items():
        if form_field in form_data:
            input_dict[feature_name] = form_data[form_field]
        else:
            # If a field is missing, raise an error or set a default
            raise KeyError(f"Form field '{form_field}' not found in input data.")

    # Add all derived features (they already have the correct keys)
    input_dict.update(derived)

    # Create DataFrame and reorder columns to match the model's expected order
    df = pd.DataFrame([input_dict])
    # Ensure we only keep columns that the model expects
    # Some feature_names might be one-hot encoded, but we only need the raw ones.
    # The preprocessor will handle encoding; we just need to provide the raw categoricals.
    # So we can filter to only those columns present in feature_names.
    # Actually, feature_names includes all original column names (raw + derived).
    # We'll reorder to match.
    try:
        df = df[feature_names]
    except KeyError as e:
        missing = set(feature_names) - set(df.columns)
        raise KeyError(f"Missing columns in input: {missing}. Expected: {feature_names}") from e

    return df

def get_recommendations(prediction, prob, data):
    """Generate recommendation text based on prediction and input data."""
    recs = []
    if prediction == 1:
        recs.append("⚠️ The child is predicted to **default** on immunization. Immediate intervention is recommended.")
        if data.get('Age_of_child', 0) > 1:
            recs.append("• Child is over 1 year old – ensure catch-up vaccinations are scheduled.")
        if data.get('Received_prenatal_care') == 'NO':
            recs.append("• No prenatal care received – encourage antenatal visits and immunization awareness.")
        if data.get('Wealth_index_quintile') in ['POOREST', 'SECOND']:
            recs.append("• Low wealth quintile – consider community outreach and free immunization services.")
    else:
        recs.append("✅ The child is predicted to **not default**. Continue routine immunizations.")
        recs.append("• Maintain regular check-ups and follow the immunization schedule.")
    recs.append(f"Prediction confidence: {prob:.2%}")
    return recs
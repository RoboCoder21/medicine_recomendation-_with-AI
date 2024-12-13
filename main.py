from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Flask app
app = Flask(__name__)

# Function to load CSV files with error handling
def load_csv(file_path):

    try:
        logging.debug(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path, encoding='ISO-8859-1')  # Change encoding as needed
    except UnicodeDecodeError:
        logging.error(f"UnicodeDecodeError for file: {file_path}, trying different encoding")
        return pd.read_csv(file_path, encoding='utf-16')  # Try another encoding if needed
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file not found
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame for any other exception



# Load datasets
sym_des_en = load_csv("dataset/symptoms_en.csv")
precautions_en = load_csv("dataset/precautions_en.csv")
workout_en = load_csv("dataset/workout_en.csv")
description_en = load_csv("dataset/description_en.csv")
medications_en = load_csv("dataset/medications_en.csv")
diets_en = load_csv("dataset/diets_en.csv")

sym_des_am = load_csv("am dataset/symtoms_am - symtoms_df (1).csv")
precautions_am = load_csv("am dataset/symtoms_am - precautions_df.csv")
workout_am = load_csv("am dataset/symtoms_am - workout_df.csv")
description_am = load_csv("am dataset/symtoms_am - description (1).csv")
medications_am = load_csv("am dataset/symtoms_am - medications.csv")
diets_am = load_csv("am dataset/symtoms_am - diets.csv")

# Load models
try:
    svc_en = pickle.load(open('models/svc_en.pkl', 'rb'))
    svc_am = pickle.load(open('models/svc_am.pkl', 'rb'))
    logging.debug("Models loaded successfully")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}")
    svc_en = svc_am = None  # Handle the absence of the models

# Custom and helper functions
def helper(dis, lang):
    if lang == 'en':
        if 'Disease' not in description_en.columns:
            logging.error("'Disease' column not found in description_en DataFrame")
            return None, None, None, None, None
        
        desc = description_en[description_en['Disease'] == dis]['Description']
        pre = precautions_en[precautions_en['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        med = medications_en[medications_en['Disease'] == dis]['Medication']
        die = diets_en[diets_en['Disease'] == dis]['Diet']
        wrkout = workout_en[workout_en['disease'] == dis]['workout']
    else:
        if 'Disease' not in description_am.columns:
            logging.error("'Disease' column not found in description_am DataFrame")
            return None, None, None, None, None
        
        desc = description_am[description_am['Disease'] == dis]['Description']
        pre = precautions_am[precautions_am['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        med = medications_am[medications_am['Disease'] == dis]['Medication']
        die = diets_am[diets_am['Disease'] == dis]['Diet']
        wrkout = workout_am[workout_am['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptoms and diseases dictionaries
symptoms_dict_en = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}
symptoms_dict_am = {
    "እከክ": 0,
    "የቆዳ_ሽፍታ": 1,
    "ኖዳል_ቆዳ_ፍንዳታዎች": 2,
    "የማያቋርጥ_ማስነጠስ": 3,
    "መንቀጥቀጥ": 4,
    "ብርድ ብርድ ማለት": 5,
    "የመገጣጠሚያ_ህመም": 6,
    "የሆድ_ህመም": 7,
    "አሲድነት": 8,
    "በምላስ_ላይ_ቁስል": 9,
    "ጡንቻ_ማባከን": 10,
    "ማስታወክ": 11,
    "ሚኪቱሪሽን ማቃጠል": 12,
    "ነጠብጣብ_ሽንት": 13,
    "ድካም": 14,
    "ክብደት_መጨመር": 15,
    "ጭንቀት": 16,
    "ቀዝቃዛ_እጅ_እና_እግር": 17,
    "የስሜት_መለዋወጥ": 18,
    "ክብደት_መቀነስ": 19,
    "እረፍት ማጣት": 20,
    "ግድየለሽነት": 21,
    "በጉሮሮ ውስጥ_ጥፍሮች": 22,
    "መደበኛ ያልሆነ_የስኳር_ደረጃ": 23,
    "ሳል": 24,
    "ከፍተኛ_ትኩሳት": 25,
    "የደነቆረ_አይኖች": 26,
    "ትንፋሽ ማጣት": 27,
    "ማላብ": 28,
    "ድርቀት": 29,
    "የምግብ አለመፈጨት ችግር": 30,
    "ራስ ምታት": 31,
    "ቢጫ_ቆዳ": 32,
    "ጨለማ_ሽንት": 33,
    "ማቅለሽለሽ": 34,
    "የምግብ ፍላጎት_ማጣት": 35,
    "ከዓይን_ጀርባ_ህመም": 36,
    "የጀርባ_ህመም": 37,
    "የሆድ ድርቀት": 38,
    "የሆድ_ህመም": 39,
    "ተቅማጥ": 40,
    "መለስተኛ_ትኩሳት": 41,
    "ቢጫ_ሽንት": 42,
    "የዓይን_ቢጫ_መቅላት": 43,
    "አጣዳፊ_የጉበት_ሽንፈት": 44,
    "ፈሳሽ_ከመጠን በላይ መጫን": 45,
    "የሆድ_እብጠት": 46,
    "ያበጡ_ሊምፍ ኖዶች": 47,
    "ማዘን": 48,
    "የደበዘዘ_እና_የተዛባ_ዕይታ": 49,
    "አክታ": 50,
    "የጉሮሮ_ብስጭት": 51,
    "የአይን_መቅላት": 52,
    "sinus_ግፊት": 53,
    "ንፍጥ_አፍንጫ": 54,
    "መጨናነቅ": 55,
    "የደረት_ህመም": 56,
    "በእግሮች ውስጥ_ደካማነት": 57,
    "ፈጣን_የልብ_ምት": 58,
    "በአንጀት_እንቅስቃሴ_ጊዜ_ህመም": 59,
    "pain_in_anal_region": 60,
    "ደም የተሞላ_ሰገራ": 61,
    "በፊንጢጣ ውስጥ መበሳጨት": 62,
    "የአንገት_ህመም": 63,
    "መፍዘዝ": 64,
    "ቁርጠት": 65,
    "መሰባበር": 66,
    "ከመጠን ያለፈ ውፍረት": 67,
    "እብጠት_እግር": 68,
    "ያበጡ_የደም_ዕቃዎች": 69,
    "የታፋ_ፊት_እና_አይኖች": 70,
    "ታይሮይድ": 71,
    "ተሰባሪ_ጥፍሮች": 72,
    "ያበጡ_እጅግ": 73,
    "ከመጠን በላይ_ረሃብ": 74,
    "ከጋብቻ ውጪ ያሉ እውቂያዎች": 75,
    "ማድረቅ_እና_ከንፈሮችን_መኮረጅ": 76,
    "የተሳሳተ_ንግግር": 77,
    "የጉልበት_ህመም": 78,
    "hip_የመገጣጠሚያ_ህመም": 79,
    "የጡንቻ_ደካማነት": 80,
    "አንገተ ደንዳና": 81,
    "እብጠት_መገጣጠሚያዎች": 82,
    "የእንቅስቃሴ_ግትርነት": 83,
    "የሚሽከረከር_እንቅስቃሴዎች": 84,
    "ሚዛን_ማጣት": 85,
    "አለመረጋጋት": 86,
    "የአንድ_አካል_ጎን_ደካማነት": 87,
    "ማሽተት_ማጣት": 88,
    "የፊኛ ምቾት ማጣት": 89,
    "የሽንት መጥፎ ሽታ": 90,
    "የማያቋርጥ_የሽንት_ስሜት": 91,
    "የጋዞች_መተላለፊያ": 92,
    "ውስጣዊ_እከክ": 93,
    "መርዛማ_መልክ_(ታይፎስ)": 94,
    "የመንፈስ ጭንቀት": 95,
    "ብስጭት": 96,
    "የጡንቻ_ህመም": 97,
    "ተቀይሯል_sensorium": 98,
    "ቀይ_ቦታዎች_በአካል_ላይ": 99,
    "የሆድ_ህመም": 100,
    "ያልተለመደ_ወር አበባ": 101,
    "dischromic _patches": 102,
    "ከዓይን_ማጠጣት": 103,
    "የምግብ ፍላጎት መጨመር": 104,
    "ፖሊዩሪያ": 105,
    "የቤተሰብ_ታሪክ": 106,
    "mucoid_sputum": 107,
    "ዝገት_አክታ": 108,
    "የማተኮር_ማጣት": 109,
    "የእይታ_ረብሻዎች": 110,
    "ደም_መውሰድ_መቀበል": 111,
    "የማይጸዳ_መርፌ መቀበል": 112,
    "ኮማ": 113,
    "የሆድ_መድማት": 114,
    "የሆድ_መረበሽ": 115,
    "የአልኮል_ፍጆታ_ታሪክ": 116,
    "ፈሳሽ_ከመጠን በላይ መጫን": 117,
    "ደም_በአክታ": 118,
    "በጥጃው_ላይ የታወቁ_ጅማቶች": 119,
    "የልብ ምቶች": 120,
    "የሚያሰቃይ_መራመድ": 121,
    "መግል_የተሞላ_ብጉር": 122,
    "ጥቁር ነጠብጣቦች": 123,
    "የሚንቀጠቀጡ": 124,
    "የቆዳ_መፋቅ": 125,
    "ብር_የሚመስል_አቧራ": 126,
    "በምስማር_ውስጥ_ትንንሽ_ጥርሶች": 127,
    "የሚያቃጥሉ_ጥፍሮች": 128,
    "አረፋ": 129,
    "በአፍንጫ_ዙሪያ_ቀይ_ቁስል": 130,
    "ቢጫ_ቅርፊት_አዝ": 131
}
diseases_list_en = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


diseases_list_am = {15: 'የፈንገስ ኢንፌክሽን', 4: 'አለርጂ', 16: 'GERD', 9: 'ሥር የሰደደ ኮሌስትሮል', 14: 'የመድሃኒት ምላሽ', 33: 'የፔፕቲክ ቁስለት በሽታ', 1: 'AIDS', 12: 'የስኳር በሽታ ', 17: 'የጨጓራ እጢ (gastroenteritis).', 6: 'ብሮንካይያል አስም', 23: 'የደም ግፊት ', 30: 'ማይግሬን', 7: 'የማኅጸን ጫፍ ስፖንዶሎሲስ', 32: 'ሽባ (የአንጎል ደም መፍሰስ)', 28: 'አገርጥቶትና', 29: 'ወባ', 8: 'የዶሮ ፐክስ', 11: 'ዴንጊ', 37: 'ታይፎይድ', 40: 'ሄፓታይተስ ኤ', 19: 'ሄፓታይተስ ቢ', 20: 'ሄፓታይተስ ሲ', 21: 'ሄፓታይተስ ዲ', 22: 'ሄፓታይተስ ኢ', 3: 'የአልኮል ሄፓታይተስ', 36: 'የሳንባ ነቀርሳ በሽታ', 10: 'የጋራ ቅዝቃዜ', 34: 'የሳንባ ምች', 13: 'ዲሞርፊክ ሄሞሮይድስ (ክምር)', 18: 'የልብ ድካም', 39: 'የ varicose ደም መላሽ ቧንቧዎች', 26: 'Hypothyroidism', 24: 'ሃይፖታይሮዲዝም', 25: 'ሃይፖግላይሴሚያ', 31: 'የአርትሮሲስ በሽታ', 5: 'አርትራይተስ', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'ብጉር', 38: 'የሽንት ቧንቧ ኢንፌክሽን', 35: 'Psoriasis', 27: 'ኢምፔቲጎ'}


def get_predicted_value(patient_symptoms, lang):
    if lang == 'en':
        input_vector = np.zeros(len(symptoms_dict_en))
        symptoms_dict = symptoms_dict_en
    else:
        input_vector = np.zeros(len(symptoms_dict_am))
        symptoms_dict = symptoms_dict_am

    for item in patient_symptoms:
        if item in symptoms_dict:
            index = symptoms_dict[item]
            logging.debug(f"Setting input_vector[{index}] to 1 for symptom: {item}")
            input_vector[index] = 1  # Set the appropriate index
        else:
            logging.warning(f"Symptom '{item}' not found in symptoms_dict")

    model = svc_en if lang == 'en' else svc_am

    # Check if the model was loaded successfully
    if model is None:
        logging.error("Model is not loaded. Please check the model file paths.")
        return "Model not loaded"  # Handle appropriately
    
    try:
        disease_index = model.predict([input_vector])[0]
        return diseases_list_en[disease_index] if lang == 'en' else diseases_list_am[disease_index]
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return "Prediction error"

# Creating routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()
        lang = request.form.get('lang', 'en')
        
        if symptoms == "Symptoms" or not symptoms:
            message = "Please write symptoms or check for misspellings."
            return render_template('index.html', message=message)
        
        user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]
        predicted_disease = get_predicted_value(user_symptoms, lang)
        
        if predicted_disease in ["Model not loaded", "Prediction error"]:
            return render_template('index.html', message=predicted_disease)
        
        # Call the helper function and unpack values
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease, lang)

        # Check for None return values
        if precautions is None:
            return render_template('index.html', message="Error retrieving precautions for the disease.")

        # Check for empty precautions DataFrame
        if precautions.empty:
            my_precautions = ["No precautions available for this disease."]
        else:
            my_precautions = [precautions[col].values[0] for col in precautions.columns if col in precautions]

        return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                               my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                               workout=workout)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=True)
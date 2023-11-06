
from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle
app = Flask(__name__)
# with open(f'model\encoding_config.pkl', 'rb') as f:
#     encoding_config = pickle.load(f)
with open(f'model\o_random_forest.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)
# Load the encoding configuration
encoding_config = joblib.load('model\encoding_config3.pkl')


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get user input from the form
    user_input = {}
    for key in request.form.keys():
        user_input[key] = request.form[key]

    # Create a DataFrame with the user input
    user_data = pd.DataFrame([user_input])

    # Perform the same one-hot encoding on user input
    X_new = pd.get_dummies(user_data, columns=encoding_config['categorical_columns'])
    X_new = X_new.reindex(columns=encoding_config['column_order'], fill_value=0)

    # Make predictions using the model
    prediction = random_forest_model.predict(X_new)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

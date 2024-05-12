from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model/finalized_model.pkl', 'rb'))  # Make sure the path to your model is correct

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract features from form data
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            prediction = model.predict(final_features)
            output = prediction[0]
            
            # Map numerical prediction to a meaningful class name
            species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            predicted_species = species[output]
            result = f'The predicted iris class is {predicted_species}'
        except Exception as e:
            result = "Error in processing the prediction. Please check the input values."
        
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

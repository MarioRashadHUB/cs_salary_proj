# use flask to host the model

import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model.pkl', 'rb') as f:
    model = pickle.load(f)

# initialize the flask app
app = flask.Flask(__name__, template_folder='templates')


# set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        state_code = flask.request.form['state_code']
        python_yn = flask.request.form['python_yn']
        java_yn  = flask.request.form['java_yn']
        C_plus_plus_yn = flask.request.form['C_plus_plus_yn']
        C_sharp_yn = flask.request.form['C_sharp_yn']
        PHP_yn = flask.request.form['PHP_yn']
        swift_yn = flask.request.form['swift_yn']
        ruby_yn = flask.request.form['ruby_yn']
        javascript_yn = flask.request.form['javascript_yn']
        SQL_yn = flask.request.form['SQL_yn']
        senior_yn = flask.request.form['senior_yn']
 
      
        # Make DataFrame for model
        input_variables = pd.DataFrame([[ state_code, python_yn, java_yn, C_plus_plus_yn, C_sharp_yn, PHP_yn,
                                         swift_yn, ruby_yn, javascript_yn, SQL_yn, senior_yn]],
                                       columns=["state_code", "python_yn", "java_yn", "C_plus_plus_yn", "C_sharp_yn",
                                                "PHP_yn", "swift_yn", "ruby_yn", "javascript_yn", "SQL_yn", "senior_yn"],
                                       dtype=float,
                                       index=['input'])
        
        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
        output = float(round(prediction, 2))
        
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('index.html',
                                     original_input={'state_code':state_code,
                                                     'python_yn':python_yn,
                                                     'java_yn':java_yn,
                                                     'C_plus_plus_yn':C_plus_plus_yn,
                                                     'C_sharp_yn':C_sharp_yn,
                                                     'PHP_yn':PHP_yn,
                                                     'swift_yn':swift_yn,
                                                     'ruby_yn':ruby_yn,
                                                     'javascript_yn':javascript_yn,
                                                     'SQL_yn':SQL_yn,
                                                     'senior_yn':senior_yn},
                                     result=float(output)
                                     )
        
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request,render_template
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from datapreparation.data_prep import data_prep
from train import train_class
from predict import predict_class
import json
from werkzeug.utils import secure_filename
import os
import shutil

app=Flask(__name__)
app.config['UPLOAD_FOLDER']='raw'

api=Api(app)
data_prep_instance=data_prep()
train_instance=train_class()
predict_instance=predict_class()


with open('config.json', 'r') as f:
    data = json.load(f)

auth=HTTPBasicAuth()
users={
    "admin":generate_password_hash("okayboss")
}

@auth.verify_password
def verify_password(username,password):
    if username in users and check_password_hash(users.get(username),password):
        return username


@app.route('/')
# @auth.login_required
def index():
    return render_template('index.html', msg="I'm working")

@app.route('/data_present',  methods = ['GET', 'POST'])
def trained_data():
    listOfTrained=data_prep_instance.data_trained(data['csv_path'])
    return render_template('index.html', list_status=listOfTrained)


@app.route('/train',  methods = ['GET', 'POST'])
def train():
    return render_template('train.html') 


@app.route('/upload',  methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        folder_name = request.form['text']

    create_folder_path=app.config['UPLOAD_FOLDER'] 
    if not os.path.exists(create_folder_path):
        # create the folder
        os.mkdir(create_folder_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")

    
    for file in files:
        filename = secure_filename(file.filename)
        try:
            directory = app.config['UPLOAD_FOLDER'] +'/'+ folder_name
            os.mkdir(directory)
        except FileExistsError:
            pass
        file.save(os.path.join(directory, filename))
    data_prep_instance.process_data(data['images_dir'],data['csv_path'],data['pose_model'],data['body_dict'],'train')

   
        # check if the folder exists before attempting to delete it
    if os.path.exists(create_folder_path):
        # use shutil.rmtree() function to delete the folder and all its contents
        shutil.rmtree(create_folder_path, ignore_errors=True)
        print("Folder deleted successfully.")
    else:
        print("Folder does not exist.")

    return render_template('train.html', file_status="'Files uploaded successfully!") 


@app.route('/data_prep_fun',  methods = ['GET', 'POST'])
def data_prep_fun():
    acc=train_instance.train_model(data['csv_path'])
    return render_template('train.html', data_prep_fun_status="Model accuracy is: " + str(acc)) 


@app.route('/predict',  methods = ['GET', 'POST'])
def predict():
    return render_template('predict.html')


@app.route('/upload_predict',  methods = ['GET', 'POST'])
def upload_predict_file():
    if request.method == 'POST':
        file = request.files['file']
        folder_name = 'unknow'
    os.mkdir('upload')
    filename = secure_filename(file.filename)
    try:
        directory = 'upload' +'/'+ folder_name
        os.mkdir(directory)
    except FileExistsError:
        pass
    file.save(os.path.join(directory, filename))

    data_prep_instance.process_data(data['predict_video'],data['predict_csv'],data['pose_model'],data['body_dict'],'predict')

    if os.path.exists('upload'):
        # use shutil.rmtree() function to delete the folder and all its contents
        shutil.rmtree('upload', ignore_errors=True)
        print("Folder deleted successfully.")
    else:
        print("Folder does not exist.")

    return render_template('predict.html', file_status="Predict File uploaded successfully!") 


@app.route('/prdict_data',  methods = ['GET', 'POST'])
def prdict_data():
    result=predict_instance.predict_model(data['predict_csv'])
    return render_template('predict.html', predict_status=result)


@app.route('/backtohome', methods = ['GET', 'POST'])
def backtohome():
    return render_template('index.html')


if __name__== '__main__':
    app.run(debug=True,host='0.0.0.0',port=5002)


    

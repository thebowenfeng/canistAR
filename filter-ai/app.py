from flask import Flask, jsonify, redirect, url_for, make_response, request
from helper_functions import *
from autoencoders import *

app = Flask(__name__)

@app.route('/getresult', methods=['GET', 'POST'])
def getresult():
    image_binary = request.form.get("Image_string")
    img_array = base64_to_array(image_binary)
    result = predict_image(img_array)
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    #app.run(host='localhost')

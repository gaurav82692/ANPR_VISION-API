import os
from flask import Flask, request, render_template, send_from_directory, redirect,url_for, Response
import numpy as np
import tensorflow as tf
import cv2
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import re
from PIL import Image as im
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'indigo-listener-828-d01ad2f2c179.json'
client = vision.ImageAnnotatorClient()

classes = ["background","number plate"]

np.set_printoptions(suppress=True)

app = Flask(__name__)
#run_with_ngrok(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
 

    # Replace this with the path to your image
    folder='images'
    ex=folder+'/'+filename
    image = im.open(ex)
    img=cv2.imread(ex)
    #img = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)
    colors = np.random.uniform(0,255,size=(len(classes),3))
    with tf.io.gfile.GFile('num_plate.pb','rb') as f:
    	graph_def=tf.compat.v1.GraphDef()
    	graph_def.ParseFromString(f.read())
    with tf.compat.v1.Session() as sess:
    	sess.graph.as_default()
    	tf.import_graph_def(graph_def, name='')
    	rows=img.shape[0]
    	cols=img.shape[1]
    	inp=cv2.resize(img,(220,220))
    	inp=inp[:,:,[2,1,0]]
    	out=sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
						sess.graph.get_tensor_by_name('detection_scores:0'),
                      		sess.graph.get_tensor_by_name('detection_boxes:0'),
                      		sess.graph.get_tensor_by_name('detection_classes:0')],
                     		feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1],3)})
    	num_detections=int(out[0][0])
    	for i in range(num_detections):
    		classId = int(out[3][0][i])
    		score=float(out[1][0][i])
    		bbox=[float(v) for v in out[2][0][i]]
    		label=classes[classId]
    		if (score>0.3):
    			x=bbox[1]*cols
    			y=bbox[0]*rows
    			right=bbox[3]*cols
    			bottom=bbox[2]*rows
    			color=colors[classId]
    			cv2.rectangle(img, (int(x), int(y)), (int(right),int(bottom)), color, thickness=1)
    			crop = img[int(y):int(bottom), int(x):int(right)]
    			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    			Cropped = cv2.resize(gray,(300,100))
    			ret, thresh4 = cv2.threshold(Cropped, 120, 255, cv2.THRESH_TOZERO) 
    			success, encoded_image = cv2.imencode('.png', thresh4)
    			data = encoded_image.tobytes()
    			print(type(data))
    			image = vision.types.Image(content=data)
    			response = client.document_text_detection(image=image)
    			doc = response.full_text_annotation.text

    # display the resized image
    #image.show()

    #cv2_imshow(img)
    
    return render_template("complete_display_image.html",image_name=filename,text=doc)

    
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/go back')
def back():
    return redirect("http://codebugged.com/", code=302)

@app.route('/go again')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
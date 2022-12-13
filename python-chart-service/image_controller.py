# -*- coding: utf-8 -*-
# from flasgger import Swagger, swag_from
import json

import cv2
import numpy
from flask import Flask, request, jsonify

import engine
import utils

app = Flask(__name__)
# swagger = Swagger(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image/upload', methods=['POST'])
# @swag_from("swagger/image_controller_api_doc.yml")
def upload_image():
    if 'image' not in request.files:
        print(request.files)
        resp = jsonify({'message': 'No image in request'})
        resp.status_code = 400
        return resp
    image = request.files['image']

    if image and allowed_file(image.filename):
        success = True
    else:
        resp = jsonify({'message': 'File type is not allowed'})
        resp.status_code = 400
        return resp

    if success:
        clusters_num = request.args.get('clusters')
        rows_num = int(request.args.get('rows'))
        columns_num = int(request.args.get('columns'))

        # read image file string data
        # convert string data to numpy array
        image = image.read()
        image_bytes = numpy.fromstring(image, numpy.uint8)
        # convert numpy array to image
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        utils.showImage(image)
        labels, label_image_map = engine.get_cells_from_image(image, clusters_num, rows_num, columns_num)
        response = {}
        labels = labels.reshape(rows_num, columns_num)
        # labels_base64 = engine.encode_base64(labels)
        response["matrix"] = labels.tolist()
        response['symbolsMap'] = label_image_map

       # пока хардкодим ответ. После появления этапа тех.поддержки будем получать ответ оттуда
       #  with open('./response/image_clusterize_api_response.json') as file:
       #      response = json.load(file)

        resp = app.response_class(
           response=json.dumps(response),
           status=202,
           mimetype='application/json'
        )
        return resp

# @app.route('/fonts', methods=['GET'])
# def get_fonts_from_file():
#     font_service.get_fonts_from_file("python-chart-service/fonts/CSTITCTG.ttf")


if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')

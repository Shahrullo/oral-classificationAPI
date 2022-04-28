# Prevent ImportError w/ flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
import flask_restful
from tkinter import E
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import torch
import torchvision
# RESTful API packages
from flask_restplus import Api, Resource
from flask import Flask, jsonify
# Utility Functions
from util import oralmodel





application = app = Flask(__name__)
api = Api(app, version="1.0", title="ORAL DETECTION API", 
        description="Identifying if an image is oral or not")
ns = api.namespace(
    "ArtificialIntelligence", 
    description="Represents the image category by the AI."
)

# Use Flask-RESTPlus argparser to process user-uploaded images
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)

# tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
# model = tf.keras.models.load_model('./model_weight/simon')
model = torchvision.models.resnet50(pretrained=True)
model = oralmodel.modify_model(model)
model.load_state_dict(torch.load('./model_weight/pytorch/model-augmented.pt', map_location='cpu'))

print("Loaded model from disk")
# model.compile(loss='binary_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])


# Add the route to run inference
@ns.route("/prediction2")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser, 
             description="Let the AI predict if its oral or not.")
    def post(self):
        # A: get the image
        image = oralmodel.get_image(arg_parser)
        # B: preprocess the image
        final_image = oralmodel.preprocess_image(image)
        # C: make the prediction
        prediction = oralmodel.predict_oral(model, final_image)
        # return the classification
        return jsonify(prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8080)

    # from waitress import serve
    # serve(app, port=8080)



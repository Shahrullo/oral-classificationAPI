# Processing Input and Output Data
from PIL import Image
import torch
from torchvision import transforms

LABELS=['oral', 'nonoral']

def get_image(arg_parser):
    '''Returns a Pillow Image given the uploaded image.'''
    args = arg_parser.parse_args()
    image_file = args.image  # reading args from file
    return Image.open(image_file).convert('RGB')  # open the image


def preprocess_image(image):
    """Converts a PIL.Image into a Tensor of the 
    right dimensions 
    """
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = test_transforms(image).unsqueeze_(0)
    # size=(150,150)
    # image_data=ImageOps.fit(image, size)
    # image_data= np.asarray(image_data)
    # new_image= cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    # resized_image= cv2.resize(new_image, (256,256))
    # final_image=np.reshape(resized_image, [1,256,256,3])
    
    return image


def predict_oral(model, image):
    """Returns the most likely class for the image 
    according to the output of the model.

    Parameters: model and image

    Source: https://tinyurl.com/dzav422a

    Returns: dict: the label-ORAL or NONORAL and the models confidence-percentage of correct prediction associated thereof it
                   are included as fields
    """
    model.eval()
    prediction_probabilities = model(image)
    # get the prediction label
    index_highest_proba = torch.max(prediction_probabilities, 1)[1]
    label = str(LABELS[index_highest_proba])
    # get the prediction prob
    # confidence = float(100*np.max(prediction_probabilities))
    # return the output as a JSON string
    output = {
        "Image captured is": label, 
        #  "confidence": confidence,
    }

    # prediction_probabilities = model.run(image)
    # # get the prediction label
    # index_highest_proba = np.argmax(prediction_probabilities)
    # label = str(LABELS[index_highest_proba])
    # # get the prediction probability
    # confidence = float(100*np.max(prediction_probabilities))
    # # return the output as a JSON string
    # output = {
    #     "Image captured is": label, 
    #      "confidence": confidence,
         
    # }
    return output


def modify_model(model):

    IN_FEATURES = model.fc.in_features
    OUTPUT_DIM = 2

    fc = torch.nn.Linear(IN_FEATURES, OUTPUT_DIM)

    model.fc = fc

    return model
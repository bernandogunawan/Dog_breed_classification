### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['Chihuahua','Japanese_spaniel','Maltese_dog','Pekinese','Shih-Tzu',
 'Blenheim_spaniel','papillon','toy_terrier','Rhodesian_ridgeback','Afghan_hound','basset',
 'beagle','bloodhound','bluetick','black-and-tan_coonhound','Walker_hound',
 'English_foxhound','redbone','borzoi','Irish_wolfhound','Italian_greyhound',
 'whippet','Ibizan_hound','Norwegian_elkhound','otterhound','Saluki',
 'Scottish_deerhound','Weimaraner','Staffordshire_bullterrier','American_Staffordshire_terrier','Bedlington_terrier',
 'Border_terrier','Kerry_blue_terrier','Irish_terrier','Norfolk_terrier','Norwich_terrier',
 'Yorkshire_terrier','wire-haired_fox_terrier','Lakeland_terrier','Sealyham_terrier',
 'Airedale','cairn','Australian_terrier','Dandie_Dinmont','Boston_bull',
 'miniature_schnauzer','giant_schnauzer','standard_schnauzer','Scotch_terrier',
 'Tibetan_terrier','silky_terrier','soft-coated_wheaten_terrier','West_Highland_white_terrier',
 'Lhasa','flat-coated_retriever','curly-coated_retriever','golden_retriever',
 'Labrador_retriever','Chesapeake_Bay_retriever','German_short-haired_pointer',
 'vizsla','English_setter','Irish_setter','Gordon_setter','Brittany_spaniel',
 'clumber','English_springer','Welsh_springer_spaniel','cocker_spaniel',
 'Sussex_spaniel','Irish_water_spaniel','kuvasz','schipperke','groenendael',
 'malinois','briard','kelpie','komondor','Old_English_sheepdog','Shetland_sheepdog',
 'collie','Border_collie','Bouvier_des_Flandres','Rottweiler','German_shepherd','Doberman',
 'miniature_pinscher','Greater_Swiss_Mountain_dog','Bernese_mountain_dog',
 'Appenzeller','EntleBucher','boxer','bull_mastiff','Tibetan_mastiff','French_bulldog',
 'Great_Dane','Saint_Bernard','Eskimo_dog','malamute','Siberian_husky','affenpinscher',
 'basenji','pug','Leonberg','Newfoundland','Great_Pyrenees','Samoyed',
 'Pomeranian','chow','keeshond','Brabancon_griffon','Pembroke','Cardigan',
 'toy_poodle','miniature_poodle','standard_poodle','Mexican_hairless',
 'dingo','dhole','African_hunting_dog']

### 2. Model and transforms preparation ###

# Create EffNetB2 model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names), # len(class_names) would also work
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="best_model_efficientnet_b2.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img):
    
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Dog Breed Classification"
description = "An EfficientNetB2 feature extractor computer vision model to classify dog breeds"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
demo.launch()

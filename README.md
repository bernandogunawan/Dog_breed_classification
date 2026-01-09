# Dog Breed Classification

This project focuses on building and evaluating deep learning models for **dog breed classification from images**.  
It covers the entire experimentation and training pipeline, including data preparation, model development, transfer learning, and evaluation.


## Project Goals
The main goals of this project are to:
- Train deep learning models to classify dog breeds from images
- Experiment with different pretrained architectures using transfer learning
- Evaluate model performance and select the best-performing model
- Maintain a modular and reusable PyTorch codebase


## Dataset
The dataset are from [Stanford dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/). It consists of labeled images of dogs across multiple breeds.

- Each class corresponds to a unique dog breed
- Images vary in size, pose, lighting, and background

All dataset loading and preprocessing logic is implemented in `src/data_setup.py`.


## Model
This project uses **transfer learning** with pretrained convolutional neural networks.

### Architectures Experimented With
- **ResNet50**
- **EfficientNet-B2**

### Final Model Selection
- **EfficientNet-B2** was selected as the best-performing model based on evaluation metrics on the test dataset
- Pretrained ImageNet weights were fine-tuned for dog breed classification

Model construction logic is implemented in `src/model_builder.py`.

## Training and Evaluation
### Training
- Training loops are implemented using PyTorch
- Training logic is centralized in `src/engine.py`

### Evaluation
- Model performance is evaluated on a held-out test set
- **Accuracy** is used as the primary evaluation metric
- Evaluation results are used to select the best model checkpoint


## Experiment Tracking
Experiments are tracked with `comet ml` for comparison between model variants.

- Multiple model architectures and configurations are tested
- Training runs can be executed via scripts or notebooks
- Experiment results guide model selection and improvement

Notebook-based experimentation is located in the `notebooks/` directory.

## Installation
Follow the steps below to setup project environment

1.  clone the repository
```bash
git clone https://github.com/bernandogunawan/Dog-breed-classification.git
cd Dog-breed-classification
```
2.  Create virtual environment
```bash
python -m venv venv
source venv/binactivate      # Linux / macOS
venv\Scripts\activate        # Windows
```
3.  Install dependencies
```bash
pip install -r requirements.txt
```

## Running Experiment
To start experiment, run:
```bash
python notebooks/run_experiment.py
```

## Results
- EfficientNet-B2 achieved the best accuracy on the test dataset
- The best model checkpoint is saved for later inference and deployment

## Future Work

Potential extensions to this project include:
- Additional evaluation metrics (precision, recall, F1-score)
- Confusion matrix and per-class performance analysis
- Try more model to experiment
- Deployment via web or API interface
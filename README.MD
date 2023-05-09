
# Tinder Machine Learning
![WORK IN PROGRESS](https://img.shields.io/badge/WORK%20IN%20PROGRESS-YES-GREEN?style=for-the-badge)
![STATUS](https://img.shields.io/badge/STATUS-working-green?style=for-the-badge)
![Completion](https://img.shields.io/badge/COMPLETION-75%25-orange?style=for-the-badge)
![MADE WITH](https://img.shields.io/badge/MADE%20WITH-PYTHON-blue?style=for-the-badge)
![MADE WITH](https://img.shields.io/badge/MADE%20WITH-SELENIUM-blue?style=for-the-badge)
![MADE WITH](https://img.shields.io/badge/MADE%20WITH-KERAS-blue?style=for-the-badge)
![MADE WITH](https://img.shields.io/badge/MADE%20WITH-TENSORFLOW-blue?style=for-the-badge)
![MADE WITH](https://img.shields.io/badge/MADE%20WITH-BS4-blue?style=for-the-badge)

Currently Absolutely broken as I overhual it with my new found experience with machine learning and overall programming this wonderful "research" tool will be rolled out SOON.
This is soley for "research" purposes for ultilizing image classification with keras base models, google resnet and inception models.
current highest accuracy model has achieved is 85% and the model/weights will not be uploaded.

Please read tinder tos before using this script, use at your own discretion.   


## Installation
Below are the libraries required to run the script
```python
  pip install -r requirements.txt
```
    
## Model implementation Roadmap

- Implement Google Resnet [X]

- Implement Inception v3 [X] 

- Implement GPT-3 [O] - due to the nature of the model it is not suitable for this task

  
## Running Instructions

STEP 1 - Get your data

"python tinder.py mode=data", this will launch selenium firefox/whichever browser you set it to run with and continously store your likes/dislikes. This requires your browser to be already logged into tinder in the browser profile you provide.


STEP 2 -  Train on your gathered data

"python tinder_model.py", Once you have gathered n number of images you can start attempting to train your model. my epoch count is quite ridculous but I am training on 6 GPUs.

STEP 3 - Test Validate 

"python tinder.py mode=auto" , This will  start "research" swiping based on the models prediction on what you find attractive.

STEP 4 - Repeat entire process until you're happy or give up and embrace the cold embrace of loneliness.

  

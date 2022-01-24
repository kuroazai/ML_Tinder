
# Tinder Machine Learning

This is soley for "research" purposes for ultilizing image classification with keras base models, google resnet and inception models.
current highest accuracy model has achieved is 85% and the model/weights will not be uploaded.

Please read tinder tos before using this script, use at your own discretion.   


## Installation
Below are the libraries required to run the script
```python
  pip install beautifulsoup4
  pip install selenium
  pip install numpy 
  pip install tensorflow
  pip install keras 
  pip install cssutils
  pip install keyboard
```
    
## Model implementation Roadmap

- Implement Google Resnet 

- Implement Inception v4

- Implement GPT-3

  
## Running Instructions

STEP 1 - Get your data

"python tinder.py mode=data", this will launch selenium firefox/whichever browser you set it to run with and continously store your likes/dislikes. This requires your browser to be already logged into tinder in the browser profile you provide.


STEP 2 -  Train on your gathered data

"python tinder_model.py", Once you have gathered n number of images you can start attempting to train your model. my epoch count is quite ridculous but I am training on 6 GPUs.

STEP 3 - Test Validate 

"python tinder.py mode=auto" , This will  start "research" swiping based on the models prediction on what you find attractive.

STEP 4 - Repeat entire process until you're happy
## Credits

- [@geeksforgeeks](https://www.geeksforgeeks.org/python-image-classification-using-keras/)

  

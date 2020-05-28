# Hand Gesture Detector

## Requirements
- `pytorch`
- `torchvision`
- `python`
- `OpenCV`

## Description
This repo contains the code and dataset used to train a hand sign detector using PyTorch, to detect three signs to control a media player - Play/Pause, Next and Previous

## Files
- Dataset - Located in the Dataset folder as .jpg images separated by folders corresponding to their classes/hand-signs (Arrow Left, Arrow Right, Stop)
- `train.py` - Used to train the network
- `test.py` - Used to evaluate test scores of the trained network 
- `Network.py` - Contains the Neural Network representation
- `data_loader.py` - Contains useful functions to convert dataset into csv
- `camera.py` - Captures camera feed and displays the output of the neural network on the feed

## Steps for execution
- Convert the dataset into csv by executing,

      python data_loader.py

- You can alter the layers of the neural network in `Network.py` and execute the following to train the model,

      python train.py
      
- Evaluate the model using the test dataset generated by `dataloader.py` by executing,

      python test.py

- See the model in action by executing,

      python camera.py

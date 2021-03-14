## Customer response task

### How to run

- Create a virtual environment
- Install the requirements  
    'pip install requirements.txt'
- Training  
    'python train.py' #runs all models one after other
    optional args:
    - '--path': Path to training data. Default: `./Recruiting_Task_InputData.csv`
    - 'python train.py Random': to run random forest model. Default: `all`. Available options: One of [`Random`, `Logistic`, `SVM`]
    - program creates a model folder and saves all model files.

- Testing
    'python tester.py' : to run testing file
    -path to model file should be 

- to run neuralNetwork.py filw:
	'python neural_network.py' 


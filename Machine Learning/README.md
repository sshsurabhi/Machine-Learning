# Machine Learning Techniques to check the Response of People

This is objectively I can say a simple Regression task but somewhat different from normal machine learning tasks and which I completed in a while but i was very curious about the task when I compare the results with other models.
This task analyses different people activies based on their individual activities that results in their opinion.

**In the down side I have given all the directions to install the dependencies and to run the code**


The repository includes:
* Source code to clean the data and prepare the code to train.
* Training code to built SVM, KNN, Logistic Regression, Random Forest models.
* There is a seperate training code for neural network training using keras.
* Resulting visualizations will be automatically saved in to a separtae folder.
* No GPU is needed to run the model.
* Trained model file willl be saved in to the folder and it will be used to test the model in later stages.
* An excel sheet having some basic data to train the model.

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


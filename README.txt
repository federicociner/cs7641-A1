## CS7641 Assignment 1
## Federico Ciner (fciner3)
## Spring 2018

This project contains all of the necessary code and data to run the experiments used in CS7641 Assignment 1.

## Setup

1. Ensure you have Python 2.7 installed (any version above 2.7.10 will work) on your system, along with all standard libraries including the 'pip' and 'setuptools' packages.

2. Run 'pip install -r requirements.txt' to install the necessary Python libraries.

## Running the experiments

All of the necessary code to run the experiments, train the models and generate all evaluation metrics/plots is located in the 'src' folder.

1. To perform data pre-processing and generate the histogram and correlation matrix plots for both datasets, run the main() functions in the 'data_preprocess.py' and 'data_graphs.py' modules. The processed data sets will be written to the 'data/experiments' folder, and the plots will be written to the 'plots/datasets' folder.

2, To train the learners on the two datasets, run the main() function in the 'model_train.py' module. This will produce the pickled model files and cross-validation results, which will be saved in the 'models/<learner>' folder, where <learner> is the name/type of learner being trained (e.g. KNN, decision tree). Three files will be produced per dataset, per learner - a pickled GridSearchCV object, a pickled "best_estimator" model, and a CSV containing cross-validation results. Therefore, for each learner there should be 6 generated files in total (3 per dataset).

3. To generate performance measures, including validation, iteration, timing, and learning curves, run the main() function in the 'model_evaluation.py' module. This will output various .png plots and .csv result tables in the 'plots/<learner>' and 'results/<learner>' folders, where <learner> is the name/type of learner being trained.

## Credits and References
All implementations used for this assignment were taken directly from the Python scikit-learn library (http://scikit-learn.org).

Python code for graphs, model training and model evaluation adapted from Jonathan Tay (https://github.com/JonathanTay) and Adrian Chang (https://github.com/amchang).



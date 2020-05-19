# An Anomaly Detection-based React and Respond System for Trusted Cloud Platforms
Companion Repo
by Eugene Lin - Christ Church, University of Oxford.
Paper not linked as of yet...

# ###################
 In this repo you will find the implementation of methodologies described in the author's thesis.

The main goal of the code in this repo is to facilitate hyperparameter optimisations to be run over the problem space.

To do this, access hyperopt_eval.py for the first loss function (path embedding loss) as defined in the author's thesis, and validation_functions.py for the second (anomaly detection loss).

To configure the search space, alter the 'space' variable in the main function.




Code is slightly rough, but comments are included in function definitions.




List of Dependencies (version Author utilised)

Python 3.6,
numpy (1.18.1),
pandas (0.25.3),
tensorflow (1.15.0),
tensorflow-gpu (1.15.0),
keras (2.2.4),
keras-gpu (2.2.4),
scikit-learn (0.22.1),
scipy (1.4.1),
seaborn (0.9.0),
umap-learn (0.3.10),
matplotlib (3.1.1),
hyperopt (0.2.2),
nltk (3.4.5),
gensim (3.8.0),
plotly (4.6.0),

# Ticket Priority Classification

### Dataset

- For this feature, we have used the ***short descriptions*** and ***priority*** columns available in the given ticket dumps.

- There are 7 ticket dumps in the form of excel files, out of which 3 files contains the information about the ***short descriptions*** and ***priority*** required to built the priotiy classification model.

- After combining these 3 files, we have around 164,000 datapoints. Which contains following 17 categories,
	
	***'1 - Critical', '1 - Urgent', '2 - High', 
	'2 - Normal', '3 - Moderate', '4 - Low', 
	'5 - Minor', 'CRITICAL', 'HIGH', 
	'LOW', 'MEDIUM', 'PROJECT', 
	'Priority 1', 'Priority 2', 'Priority 3',
    'Priority 4', 'Service Request'***

- We re-arranged these categories into ***'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'*** as follows,

	- ***'4 - Low', '5 - Minor' --> 'LOW'***

    - ***'3 - Moderate', '2 - Normal', 'Priority 4' --> 'MEDIUM'***

    - ***'Priority 3' --> 'HIGH'***
    
    - ***'Priority 1', 'Priority 2', '1 - Critical', '1 - Urgent', '2 - High' --> 'CRITICAL'***

  and dropped ***'PROJECT'*** and ***'Service Request'*** as they are not required for our model.

- To balance this dataset, we used an over sampling technique called **SMOTE (Synthetic Minority Oversampling)**, which is based upon KNN algorithm.

<hr>

### Preprocessing pipeline

- The preprocessing pipeline is built using sklearn pipelines module.

- In order to build the model, we have to clean and vectorize the ***short descriptions*** as we are using it as the feature. The steps for preprocessing are,
	
	- Lowercasing the text

	- Removing punctuations and words containing numbers

	- Removing error charecters from the text

	- Removing multiple space from the text

	- Removing short descriptions with less than 2 words

<hr>

### Training pipeline

- The training pipeline is built using imbalanced-learn pipelines module.

- Due to the limitations of partial fit method available in only a few algorithms, we have to train the models from scratch every time we'll want to update the model.

- In order to vectorize the ***short descriptions*** text, due to the size of sparse vectors generated from Counte vectorizers and TfIdf vectorizers, we've used Hashing vectorizer in order to save the memory in production environment.

- We have devided our dataset into 80% training, 10% testing and 10% validation sets.

- This pipeline fits multiple classification algorithms and gives us the best one based on F1 score of the fitted model. Due to this our approach is dependent on the nature of data and not a specific classification algorithm.

- Based on the currently available data, the best model we have is RandomForest with default hyperparameters which is giving us around 78% accuracy score and 79% F1 score.

<hr>
# Real-Estate-Recommendation

The Project focusses on building a model which would be able to predict the SalePrice of the flat given the features to the model and also recommend some of the properties that would be suitable accrding to given features. The project can be used in the field of real estate.

The Tech stack that is involve while building the project is:
1. Python
2. Streamlit
3. Mlflow
4. Docker
5. AWS EC2

The Dataset that was used to build the projects was collected by different real estate agents with the help of domain experts. Following are the features that were considered: 'builderName', 'State', 'District', 'Location', 'Highway', 'Railway', 'Metro', 'busStop', 'closestEducationalInstituteDistance', 'closestHospitalDistance','closestMallDistance', 'closestBusinessParkDistance', 'flatNumber', 'flatType', 'Area', 'noOfBathrooms', 'Parking', 'pricePerSqFeet', 'totalPrice', 'Gym', 'swimmingPool', 'Garden', 'yogaArea', 'playArea', 'shoppingArea', 'ATM/Finance'.

Execution of the project was done by follwoing steps:
1. The First step was to properly understand the problem statement and check if the dataset required is sufficient or not then choose the tech stack that would be used in the project.
2. The Second step was clean the data then analyze the dataset using EDA and check for the relationships betweeen different features with target.
3. Then find out the features which would be necessary to predict the target variable.
4. After that we do some pre-processing required on the dataset for training our model.
5. After pre-processing we split the dataset into train and test for testing different Machine Learning algorithms.
6. The ML Algorithms that I have used in the project are "LinearRegression", "RandomForest", "GradientBoost", "SVR" and also used ensemble learning technique "Stacking" and compare the metrics also save them for predicting the SalePrice.
7. For tracking the model I have used MLflow and log the metrics of different Algorithms and compare them.
8. After I had use K-Means clustering for recommending the results to the user by segaragating different observations into results and save the model.
9. After that I have made an app Streamlit for better user interaction combining the results both the saved models.
10. In the last step I have build the docker image and push it to hub.



# HEART DISEASE PREDICTION 

## Overview

Heart disease refers to a range of conditions affecting the heart. The most common form is coronary artery disease (CAD), which involves narrowing or blockage of the coronary arteries, often leading to heart attacks. Common types of heart disease include Coronary Artery Disease (CAD) – Narrowed arteries reduce blood flow to the heart, heart arrhythmias – Irregular heartbeats, congenital heart defects – heart structure problems present at birth, cardiomyopathy – disease of the heart muscle, heart failure – The heart doesn't pump blood as well as it should, heart valvular disease – Valves don't open or close properly. etc Common risk factors are smoking, age,persistent high blood pressure, family history, high cholesterol, Sex (male higher risk), obesity, ethnicity (for example, South Asians at higher risk), diabetes, lack of physical activity and many more.	

This project uses a real world clinical datasets from UCI Library to predict the presence of heart disease in patients using Exploratory Data Analysis (EDA), Hypothesis Testing and Logistic Regression Model. The workflow is to perform an ETL (Extract, Transform, Load), create visualizations using Python and associated libraries such as matplotlib, seaborn for static visualizations and plotly was used to create interactive visualizations.  Hypothesis Testing and Logistic regression Model to predict a Target Variable (Presence of Heart Disease) among various features or key diagnostic metrics.

## Project Objectives

- Understand the relationships between various clinical features and heart disease.
- Perform statistical hypothesis testing to validate significant variables
- Build and evaluate a logistic regression model, Random Forest to predict the likelihood of a heart disease.

## Workflow and Tasks

- **Extract** data from a CSV file
- **Transform** the data using cleaning and preprocessing techniques
- **Load** the data into a pandas DataFrame for analysis
- Perform **exploratory data analysis (EDA)**
- Perform Hypothesis Testing
- Create visualizations using **matplotlib**, **seaborn** and **plotly**
- Export cleaned heart_disease datasets to Power BI
- Create 4 advanced visualizations and Dashboard on Power BI

## Technologies Used

- Python 3.12.8
- pandas
- matplotlib
- seaborn
- plotly
- ScikitLearn
- NumPy
- Power BI
- Jupyter Notebook
- Git & GitHub
- vs code
## Datasets

The dataset used is a CSV file containing clinical data from UCI Heart Disease Library. The original data was collected from 4 Locations:
 - 1. Cleveland Clinic Foundation (cleveland.data)
 - 2. Hungarian Institute of Cardiology, Budapest (hungarian.data)
 - 3. V.A. Medical Center, Long Beach, CA (long-beach-va.data)
 - 4. University Hospital, Zurich, Switzerland (switzerland.data)

The original databases have 76 raw attributes or features, only 14 of them are actually used and i only used 13 of the features aside the target variable. Thus I've taken the liberty of making 2 copies of each datasets and loaded them on jupiter notebook. The authors of the datasets have made request that any publications resulting from the use of the data include the names of the principal investigator responsible for the data collection at each institution.  They would be:
 - 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
 - 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
 - 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
 - 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
	  Robert Detrano, M.D., Ph.D.

- **Source**: [UCI Heart Disease Dataset] (https://archive.ics.uci.edu/dataset/45/heart+disease) or from Kaggle equivalent.
- **Target Variable**: 'target' -- 1 (heart disease present, 0(no heart disease)

- **Clinical Features**
Data columns (total 14 columns):
 # Column                Datatype              Description
  
 - 1  age                numeric                Age in Years
 - 2  sex                category (binary)      1 = Male 0 = Female
 - 3  chest pain type    category (Ordinal)     4 Values(-- 1 = typical angina, 
                                                         -- 2 = atypical angina, 
                                                          -- 3 = non-anginal pain, 
                                                         -- 4 = asymptomatic pain)
 - 4 resting systolic bp  numeric               Resting blood pressure (in mm Hg on admission to the hospital
 - 5 cholesterol          numeric               serum cholesterol in mg/dl

 - 6 fasting blood sugar  category (binary)          fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
 - 7 resting ECG          category (ordinal)             3 Values (resting electrocardiographic results 
                                                0 = normal,                                                       1 = having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >0.05mV)
                                                2 = showing probable or definite LVH

 - 8 max heart rate        numeric            maximum heart rate achieved
 - 9  exercise angina      category (binary)  exercise induced angina (1 = yes; 0 = no)
 - 10 ST depression        category           ST depression induced by exercise relative to rest 
                                                 (1 = present, 0 = not present)
 - 11 ST slope             category           the slope of the peak exercise ST segment, 3 Values
                                               (1 = up-sloping, 2 = flat or normal, 3 = down-sloping)
 - 12 hypertension         category           hypertension (1 = hypertensive, 0 = non-hypertensive)
 - 13 family history       category           family history of coronary artery disease (1 = yes; 0 = no)
 - 14 target               category           heart disease (1 = presence, 0 = absence)




## Key Findings
- 1. There is a significant difference in cholesterol levels between people with and without heart disease
- 2. There is a significant association of sex with heart disease with Males at increased risk
- 3. There is a significant difference in cholesterol levels across different chest pain types. 
- 4. There is a significant linear correlation between Age and Cholesterol Levels and the Age Distribution chart show a strong correlation between Heart Diseases and Cholesterol Level among certain Age groups. 
- 5. Logistic Regression produced an accuracy of 82.58% for heart disease prediction
- 6. Random Forest produced an accuracy of 83.71% for heart disease prediction

## Project Structure

- 1. Open heart_disease_prediction.ipynb` in Jupyter Notebook in VS Code IDE.
- 2. Run the notebook cells to follow the ETL workflow, visualizations and ML Algorithms
- 3. Open the heart_disease_profiling_report.html to view the ydata profiling report
- 4. Run `streamlit run app.py` to launch the interactive dashboard and prediction app.


## Key Steps

1. **Extract:** Load the CSV data using pandas.
2. **Transform:** Clean missing values, check data types, and preprocess as needed.
3. **Load:** Store the cleaned data in a DataFrame.
4. **Visualize:** Generate plots to explore relationships and trends in the cleaned datasets.

## Agile Methodothology
This is a collaborative and sustainable way of managing projects by breaking the work into small, manageable tasks, delivering results step by step, and improving tasks based on regular feedbacks (Beck, K., et al.(2001). Manifesto for Agile Software Development. [https://agilemanifesto.org]

## User Story
As a student Data Analyst, I want to build an ETL pipeline to extract, clean, transform, and load heart_disease_prediction datasets (including clinical features such as age, sex, chest pain type, resting systolic blood pressure, cholesterol level, fasting blood sugar, resting ECG, maximum heart rate during exercise, exercise angina (exercise induced chest pain), ST depression, ST slope (electrical abnormalities viewed through the ECG; a heart function test ), Hypertension, family history of coronary artery disease.

## Tasks

1. **Extract Data**  
   - Download and extract heart_disease csv datasets from Kaggle (CSV format) [(https://www.kaggle.com/cherngs/heart-disease-cleveland-uci and UCI Library https://archive.ics.uci.edu/dataset/45/heart+disease
2. **Clean Data**  
   - Handle missing values (impute or drop as appropriate).
   - Normalize data types and ensure correct formatting of all fields.
3. **Transform Data**  
   - Encode categorical variables for analytics.
4. **Load Data**  
   - Load the cleaned and transformed data into a Pandas DataFrame.
5. **Validate & Visualize Data**  
   - Check data quality and consistency (duplicates, nulls, ranges).
   - Perform descriptive statistics and correlation analysis.
   - Visualize insights and predictive models

## Acceptance Criteria

### 1. Data Extraction
- 1 The pipeline shall extract raw data from Kaggle and UCI repository in CSV format, including features: age, sex, chest pain type, resting systolic bp, cholesterol, fasting blood sugar, resting ECG, maximum heart rate during exercise, exercise angina, ST depression, ST slope, hypertension, family history of coronary artery disease and the target variabe of heart disease presence or absence.
- 2 The extraction process must log missing or duplicate records and report extraction success/failure.

### 2. Data Transformation
- 1 The pipeline shall clean data by removing duplicates, handling missing values, and standardizing field formats.
- 2 Categorical variables must be encoded for downstream analytics.

### 3. Data Loading
- 1 The data shall be loaded into a Pandas DataFrame following a pre-defined schema.
- 2 The loading process must be IDE potent, preventing duplicate entries.
- 3 The pipeline must validate and log each load, including row counts and data checksums.

### 4. Data Quality & Integrity
- 1 The pipeline shall include automated data quality checks (e.g., null checks, referential integrity) and alert on anomalies.
- 2 All transformation steps must be documented and auditable.
- 
### 5. Security & Compliance
- 1 Sensitive data must be handled in compliance with data privacy regulations (e.g., HIPAA, GDPR).
- 2 Access to data must be restricted and audit-logged.

### 6. Documentation & Monitoring
- 1 The ETL process shall be documented with setup, configuration, and troubleshooting instructions.
- 2 The pipeline must include monitoring and alerting for failures or significant data anomalies.

## Additional Requirements for Analysis & Reporting

The pipeline must enable:
- Descriptive statistics.
- Correlation and visualization reports between clinical features and heart disease risk.
- Visualization.
- Prediction models


## Visualizations
- Countplot of chest pain types vs target variable (heart disease)
- Boxplot: Cholesterol Levels vs target variable (heart disease)
- Linear correlation between Age and Cholesterol Levels?
- y data profiling report open file with a browser

## Dashboard Design
To create my Power BI dashboard, I followed a structured process to transform clinical data into meaningful visual insights. My dashboard consists of four key visuals, each focused on exploring patterns and risk factors related to heart disease: 

- **Heart Disease and Average Cholesterol Level by Age Distribution**
I used a stacked column chart to show the distribution of heart disease cases across different age groups, alongside the average cholesterol levels. This helped identify age ranges with higher cholesterol and heart disease prevalence.

- **Heart Disease by Chest Pain Types and Sex**
I created a clustered column chart to visualize how different types of chest pain (e.g., typical angina, atypical angina) relate to heart disease cases, broken down by sex. This comparison highlighted gender-specific trends in symptom presentation.

- **Correlation Matrix of Heart Disease with Common Features**
I generated a correlation heatmap using a custom heatmap by powerviz on Power BI. It displayed the strength and direction of relationships between heart disease and variables like age, cholesterol, resting systolic blood pressure, etc. This helped identify which features were most strongly associated.

- **Common Risk Factors of Heart Disease**
I built a Donut chart with filters to highlight the most frequent risk factors (e.g., Hypertension, exercise angina, fasting blood sugar) among individuals with heart disease.

## Data Ethical Considerations

- 1. I minimized data extraction by gathering only the information necessary for the project’s objectives as the original dataset was highly encrypted to the point that the features were hidden in the datasets and i only analyzed and extracted and renamed useful features to my project.
- 2. I ensured data privacy, security through encryption, secure storage, and access control measures. Whatever sensitive information that will identify patients were either absent or removed like the social security number, names etc.
- 3. I maintained transparency and accountability by documenting data handling procedures and obtaining ethical approvals. Although the datasets are public, the data collectors requested for some considerations for data usage which i complied with.
- 4. I addressed fairness and bias by reviewing the dataset for representativeness and avoiding discriminatory analysis.
- 5. I managed data mining and sharing responsibly using data use agreements that respected patient rights and usage limits. Raw data was encrypted with no sensitive identifiable information.
- 6. I complied with all legal and regulatory standards, including HIPAA, GDPR, and institutional data policies.
- 7. I preserved data accuracy and integrity by validating the dataset and reporting findings honestly.
- 8. I acknowledged and credited the sources of my data including the collectors of the data.


## Credits
- Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
- GitHub copilot for code correction, review and explanation
- Chatgpt for code research and context.
- Professional definition for Agile methodology. Beck, K., et al.(2001). Manifesto for Agile Software Development. [https://agilemanifesto.org]
- UCI data repository


## Acknowledgements
- UCI Machine Learning Repository and authors and collectors of this datasets already credited for their remarkable and profound work
- Kaggle community and everyone in the community that i made research from on the use of this datasets
- My partner who provided support and understanding through out my journey through this project.
- My instructors and fellow bootcamp mates at code institute who have been very supportive and helpful through out my journey.






*Created as part of my Data Analytics bootcamp Capstone Project!*

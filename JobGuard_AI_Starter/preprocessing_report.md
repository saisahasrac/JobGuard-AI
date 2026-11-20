# JobGuard-AI
## Data Preprocessing and Exploratory Data Analysis Report

### 1. Project Overview
JobGuard-AI is a machine learning-based system designed to
identify fraudulent job postings and distinguish them from
genuine job opportunities.

### 2. Dataset Description
Dataset: Employment Scam Aegean Dataset (EMSCAD)

Total records: 17,880
Total columns: 18

The dataset contains job posting information including:
- Job title
- Location
- Department
- Salary range
- Company profile
- Job description
- Requirements
- Benefits
- Employment type
- Required experience
- Required education
- Industry
- Function
- Fraudulent label

### 3. Exploratory Data Analysis
The dataset was explored to understand its structure,
data quality, missing values, and class distribution.

The following analyses were performed:
- Dataset dimensions
- Column analysis
- Data types
- Missing value analysis
- Class distribution
- Job description length analysis
- Description length comparison by class

### 4. Data Preprocessing

#### Missing Value Handling
Missing values in text fields were handled to prevent
processing errors and maintain compatibility with text-based
machine learning operations.

#### Duplicate Removal
Duplicate records were identified and removed to reduce
redundant information.

#### Text Cleaning
Text data was processed using:
- HTML entity decoding
- HTML tag removal
- Whitespace normalization
- String conversion
- Missing text handling

#### Text Feature Preparation
Relevant job posting fields were combined into a single
text representation.

Field prefixes were retained to preserve information about
the source of each text field.

#### Label Validation
The target variable was converted into binary labels:

0 - Genuine job posting
1 - Fraudulent job posting

### 5. Preprocessing Results

Original records: 17,880
Processed records: 17,645
Records removed: 235

Genuine postings: 16,787
Fraudulent postings: 858

### 6. Class Imbalance

The dataset contains significantly more genuine job postings
than fraudulent job postings.

This class imbalance is an important consideration during
model evaluation.

Accuracy alone may not be sufficient to evaluate the
fraud detection model.

Precision, Recall, F1-score, and confusion matrix should
also be considered.

### 7. Output Files

EDA outputs:
- missing_values.png
- class_distribution.png
- description_length.png
- description_length_by_class.png
- eda_summary.txt

Preprocessing output:
- preprocessing_summary.txt

### 8. Conclusion

Exploratory data analysis and preprocessing were performed
to understand the dataset, handle data quality issues,
prepare text data, validate labels, and generate a structured
dataset suitable for fraudulent job posting detection.

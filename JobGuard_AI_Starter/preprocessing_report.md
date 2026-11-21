# JobGuard-AI – Data Preprocessing Report

## 1. Overview

This report describes the data inspection, exploratory data analysis (EDA), and text preparation performed for the JobGuard-AI fake job posting detection project.

The objective of preprocessing is to prepare job posting information in a consistent text format that can be used by the machine learning model to identify potentially fraudulent job postings.

## 2. Dataset Overview

- **Dataset:** Fake Job Postings (EMSCAD)
- **Total records:** 17,880
- **Total columns:** 18
- **Target variable:** `fraudulent`
- **Class 0:** Genuine job postings
- **Class 1:** Fraudulent job postings

The dataset contains job-related information such as job titles, company profiles, descriptions, requirements, benefits, employment types, experience, education, industries, functions, locations, and salary ranges.

## 3. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the structure, quality, and characteristics of the dataset.

The analysis includes:

- Identification of missing values across columns
- Examination of genuine and fraudulent job posting distributions
- Analysis of job description lengths
- Comparison of description lengths between genuine and fraudulent postings

Visualizations were generated to help understand the dataset and identify patterns relevant to fake job detection.

## 4. Text Preprocessing

The preprocessing pipeline uses the `prepare_text()` function defined in `preprocessing.py`.

The function performs the following operations:

- Selects predefined text and metadata columns.
- Adds missing columns as empty strings when they are not present in the input data.
- Replaces missing values with empty strings.
- Converts the selected column values into strings.
- Combines the selected fields into a single text representation.
- Adds column-name prefixes to preserve information about the source of each text field.

The selected columns are:

- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`
- `location`
- `salary_range`

The resulting text representation retains the source-field information, allowing the model to receive a combined representation of the job posting.

**Note:** The current model-compatible preprocessing function does not perform HTML cleaning or additional whitespace normalization.

## 5. Model Compatibility

The original preprocessing implementation is retained to maintain compatibility with the existing trained model.

Changing the text preparation process may alter the input representation provided to the model and can consequently affect predictions.

Any proposed preprocessing modifications should therefore be evaluated against the existing pipeline. If substantial changes are introduced, model retraining and evaluation may be required before integration.

## 6. Conclusion

The preprocessing module combines relevant job posting fields into a unified text representation while handling missing columns and missing values.

Exploratory data analysis provides an overview of the dataset, including missing-value patterns, class distribution, and description-length characteristics.

These preprocessing and analysis steps support the development of the JobGuard-AI fake job posting detection system while preserving compatibility with the existing model.

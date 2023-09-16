# Human Study Legal Verdict Classification Analysis

This repository contains the Python scripts used to analyse the results of a human study involving the classification of case verdicts pertaining to Article 6 (Right to a fair trial) from the ECHR (European Convention on Human Rights) based on descriptions of the case facts. Participants were tasked with making a prediction of the correct verdict after reading the 'circumstances of the case' and again after reading the 'relevant legal framework', and these predictions are compared against the actual decisions taken by the ECtHR (European Court of Human Rights). The python scripts provide detailed statistical analyses, focusing on performance and confidence metrics. Below are descriptions of each script along with usage instructions.

## Scripts Description

### s7_performance_based_analysis_v1_6.py

This script performs an analysis based on the performance metrics derived from the study data. While the script's functionality and details of the analyses performed are proprietary, it generally handles:

- Data preprocessing including null value handling and type conversions.
- Performing various statistical analyses to extract insights from the data.
- Visual representation of the analyses in the form of scatter plots.

#### Usage

To use the script, execute it in a Python environment where necessary libraries are installed. The script can be configured to perform different analyses based on user input.

### s8_confidence_based_analysis_v1_1.py

This script focuses on analyzing the confidence metrics gathered during the study. The functionalities it offers include:

- Reading and preprocessing survey response data.
- Mapping categorical responses to numerical values to facilitate analysis.
- Statistical analyses to understand different confidence aspects such as early and final confidence, influence of models, and domain knowledge on confidence levels.
- Pairwise comparisons using Tukey HSD test among others.

#### Usage

Execute the script in a Python environment where the necessary libraries (mentioned at the beginning of the script) are installed. The script can be configured to perform different analyses based on user input.

## Data Files

The necessary participant classification data files for running the scripts are found in the 'Analysis' subdirectory. The model groups' final quiz responses are found in model_2nd_quiz_responses.xlsx, and the debrief survey responses for all participants are found in survey_responses.xlsx. Please ensure to maintain the structure of the repository for the scripts to function correctly.

## Installation

Ensure you have a Python environment set up with the following libraries installed:

- matplotlib
- numpy
- pandas
- scipy
- statsmodels

You can install the required packages using the following command:

```sh
pip install matplotlib numpy pandas scipy statsmodels
```


## Authors

Jack Mumford

## Acknowledgements

A big thank you to all participants of the study, whose valuable input made this analysis possible.

## Contact

For any questions or support, please contact Dr Jack Mumford at jack [dot] mumford [at] liverpool [dot] ac [dot] uk

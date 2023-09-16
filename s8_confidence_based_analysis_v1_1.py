"""
Created on Tue Jun 13 11:29:20 2023

@author: Truffles

This script processes the survey responses of the various participants and analyses
confidence-based responses.

Change log:
v1_1 = Added in main function call for user input.
v1_0 = Functional code for all groups, model/no-model, and levels of domain knowledge.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as stats
import statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


"""
def domain_analysis calculates statistics comparing the with model
and no model groups.
"""
def domain_analysis(survey_df):

    # Domain Knowledge Confidence Groups
    weak_df = survey_df[
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - CS') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - CS')]   
    moderate_df = survey_df[
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - Law') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - Law')]  
    strong_df = survey_df[
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - Domain') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - Domain')]
        
    # Domain Knowledge means
    print("Weak Confidence Mean:", weak_df['Combined Confidence'].mean())
    print("Moderate Confidence Mean:", moderate_df['Combined Confidence'].mean())  
    print("Strong Confidence Mean:", strong_df['Combined Confidence'].mean())
    print("Weak Confidence stdev:", weak_df['Combined Confidence'].std())
    print("Moderate Confidence stdev:", moderate_df['Combined Confidence'].std())  
    print("Strong Confidence stdev:", strong_df['Combined Confidence'].std(), '\n')
    
    # Domain Knowledge statistical tests   
    combined_data = pd.concat([weak_df['Combined Confidence'], moderate_df['Combined Confidence'], strong_df['Combined Confidence']], axis=0)  
    # Create a grouping variable to identify the source of each data point
    groups = ['Weak'] * len(weak_df) + ['Moderate'] * len(moderate_df) + ['Strong'] * len(strong_df) 
    # Create a DataFrame with the data and group information
    data_frame = pd.DataFrame({'Combined Confidence': combined_data, 'Group': groups})
    # Perform one-way ANOVA to test for overall differences between groups
    model = ols('data_frame["Combined Confidence"] ~ data_frame["Group"]', data=data_frame).fit()
    anova_table = sm.stats.anova_lm(model, typ=2) 
    # Perform Tukey HSD test for pairwise comparisons
    tukey_results = pairwise_tukeyhsd(data_frame["Combined Confidence"], data_frame["Group"])
    # Display the ANOVA table and Tukey HSD results
    print("ANOVA Table:")
    print(anova_table)
    print("\nTukey HSD Results:")
    print(tukey_results)  
    
    return


"""
def main requests user input to indicate what type of analysis is required.
"""
def main():
    analysis_type = input("Please enter the type of analysis ('timepoint', 'model', 'domain'): ").strip().lower()
    
    survey_df = read_individual()
    
    if analysis_type == 'timepoint':
        timepoint_analysis(survey_df)
    elif analysis_type == 'model':
        model_analysis(survey_df)
    elif analysis_type == 'domain':
        domain_analysis(survey_df)
    else:
        print(f"Unknown analysis type: {analysis_type}")
        print("Please enter one of the following: 'timepoint', 'model', 'domain'")


"""
def model_analysis calculates statistics comparing the with model
and no model groups.
"""
def model_analysis(survey_df):

    # Model-based Confidence Groups
    no_model_df = survey_df[
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - CS') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - Law') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Alpha - Domain')]   
    with_model_df = survey_df[
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - CS') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - Law') |
    (survey_df['1. Which research sub-group were you assigned to?'] == 'Group Omega - Domain')]
    
    # Model-based means abd statistical tests
    print("No model Confidence Mean:", no_model_df['Combined Confidence'].mean()) 
    print("With model Confidence Mean:", with_model_df['Combined Confidence'].mean())
    print("No model Confidence stdev:", no_model_df['Combined Confidence'].std()) 
    print("With model Confidence stdev:", with_model_df['Combined Confidence'].std(), '\n')
    u_statistic, mann_p_value = stats.mannwhitneyu(no_model_df['Combined Confidence'], with_model_df['Combined Confidence'], alternative='two-sided')
    print(f"P-value: {mann_p_value}")   
    
    return
    

"""
def plot_scatter plots a scatter plot of columns from two dataframes.
Expected parameters: df1[col], df2[col].
"""
def plot_scatter(data1, data2, data_type, label1, label2):
    ## Setting write directory
    write_path = 'Analysis'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    plt.figure(figsize=(8, 6))
    plt.scatter(data1, data2, alpha=0.5)
    #plt.title('Scatter Plot of ' + data_type)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.grid(True)
    plot_name = data_type + '_' + label1 +   '_' + label2 + '.png'
    plt.savefig(os.path.join(write_path, plot_name))
    plt.show()
    plt.close()
    return


"""
def read_individual takes k as input, indicating the session of the research study
from which the data is to be read (k=4 by default). Outputs processed excel files
to standardise the participants classifications, and adds the true verdicts.
analysis_type can take values in ['ind_scores', 'group_scores', 'model_anlys', 'domain_anlys', 
'conf_comp', 'circ_dome', 'kanst_comp', 'quiz_comp'].
"""
def read_individual(analysis_type=0):

    # Loading survey reponses dataframe
    survey_df = pd.read_excel('survey_responses.xlsx')
    quiz_df = pd.read_excel('model_2nd_quiz_responses.xlsx')
    
    # Confidence questions processing
    confidence_mapping = {'Very inaccurately': 0, 'Slightly inaccurately': 1, 'Roughly balanced': 2, 'Slightly accurately': 3, 'Very accurately': 4}
    # Apply the mapping to each column
    survey_df['Early Confidence'] = (survey_df['18.1. Immediately after training'].map(confidence_mapping))
    survey_df['Final Confidence'] = (survey_df['18.2. At the end of the project'].map(confidence_mapping)) 
    survey_df['Combined Confidence'] = (survey_df['Early Confidence'] + survey_df['Final Confidence']) / 2

    # General knowledge question processing
    location_answers = {'Brussels': 0, 'Luxembourg': 0, 'the Hague': 0, 'Strasbourg': 1}      
    polity_answers = {True: 0, False: 1}
    # Apply the mapping to each column and sum the results
    survey_df['Kanstatsin Knowledge'] = (survey_df['4. Where is the European Court of Human Rights located?'].map(location_answers) + survey_df['5. The European Court of Human Rights is the tribunal created by the United Nations.'].map(polity_answers) + survey_df['6. The European Court of Human Rights is the tribunal created by the European Union.'].map(polity_answers))

    return survey_df
    

"""
def stats_tests performs Spearman's rank correlation test and Mann-Whitney U test
to compare participant performance in accordance with some independent variable.
"""
def stats_tests(list1, list2):
    # Calculate Spearman's rank correlation
    correlation, sp_p_value = stats.spearmanr(list1, list2)
    # Perform Mann-Whitney U test
    u_statistic, mann_p_value = stats.mannwhitneyu(list1, list2, alternative='two-sided')
    # Print results
    print(f"Spearman's rank correlation coefficient: {correlation}")
    print(f"P-value: {sp_p_value}", '\n')
    print(f"Mann-Whitney U statistic: {u_statistic}")
    print(f"P-value: {mann_p_value}")
    return


"""
def timepoint_analysis calculates statistics based on confidence in performance
at the start of task compared to at the conclusion.
"""
def timepoint_analysis(survey_df):
    # Early and Final Confidence means and statistical tests
    print("Early Confidence Mean:", survey_df['Early Confidence'].mean())
    print("Final Confidence Mean:", survey_df['Final Confidence'].mean())
    print("Early Confidence stdev:", survey_df['Early Confidence'].std())
    print("Final Confidence stdev:", survey_df['Final Confidence'].std(), '\n')
    print("Early vs Final Confidence:")
    stats_tests(survey_df['Early Confidence'], survey_df['Final Confidence'])   
    return


# Calling the main function
if __name__ == "__main__":
    main()

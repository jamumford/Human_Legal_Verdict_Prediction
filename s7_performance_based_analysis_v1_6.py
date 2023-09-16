"""
Created on Tue Jun 13 11:29:20 2023

@author: Truffles

This script processes the raw outputs of the various participants and produces performance-
based analysis and plots.

Change log:
v1_6 = Added plot functionality
v1_5 = Added functionality for comparing individual performance to quiz scores.
v1_4 = Added functionality for comparing individual performance to reported confidence.
v1_3 = Adjusted code for better modular functionality, adding def splice_analysis and def print_anlys.
v1_2 = Functional code for all groups, model/no-model, and levels of domain knowledge.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as stats
import statistics


"""
def analyse_cms takes a list of confusion matrices for a given setup
and returns the various metric scores by in turn calling the different
metric functions.
"""
def analyse_cms(confuse_mats):

    # Initialise metric lists
    micro_cm = np.zeros(4)
    macro_cm = np.zeros(4)
    macro_acc_list = []
    macro_mcc_list = []
    vio_macro_prec_list = []
    non_macro_prec_list = []
    vio_macro_recall_list = []
    non_macro_recall_list = []
    productivity_list = []

    # Iterate across the confusion matrices
    for ind_cm in confuse_mats:
        micro_cm += ind_cm
        macro_cm += (ind_cm / sum(ind_cm))
        productivity_list.append(sum(ind_cm))
        
        # Update macro accuracy and MCC
        macro_acc_list.append(get_accuracy(ind_cm))
        macro_mcc_list.append(get_mcc(ind_cm))
        
        # Update macro precisions
        temp_vio_prec, temp_non_prec = get_precision(ind_cm)
        vio_macro_prec_list.append(temp_vio_prec)
        non_macro_prec_list.append(temp_non_prec)
        
        # Update macro recall
        temp_vio_recall, temp_non_recall = get_recall(ind_cm)
        vio_macro_recall_list.append(temp_vio_recall)
        non_macro_recall_list.append(temp_non_recall)
    
    # Get micro scores
    #micro_cm /= sum(micro_cm)    
    micro_acc = get_accuracy(micro_cm)
    micro_mcc = get_mcc(micro_cm)
    vio_micro_prec, non_micro_prec = get_precision(micro_cm)
    vio_micro_recall, non_micro_recall = get_recall(micro_cm)
    
    # Get macro scores
    #macro_cm /= len(confuse_mats)
    macro_acc = sum(macro_acc_list)/len(macro_acc_list)
    macro_mcc = sum(macro_mcc_list)/len(macro_mcc_list)
    macro_vio_prec = sum(vio_macro_prec_list)/len(vio_macro_prec_list)
    macro_non_prec = sum(non_macro_prec_list)/len(non_macro_prec_list)
    macro_vio_recall = sum(vio_macro_recall_list)/len(vio_macro_recall_list)
    macro_non_recall = sum(non_macro_recall_list)/len(non_macro_recall_list)
    productivity = sum(productivity_list)/len(productivity_list)
    stdev_acc = statistics.stdev(macro_acc_list)
    stdev_mcc = statistics.stdev(macro_mcc_list)
    stdev_vio_recall = statistics.stdev(vio_macro_recall_list)
    stdev_non_recall = statistics.stdev(non_macro_recall_list)
    stdev_productivity = statistics.stdev(productivity_list)
    
    print("Mirco CM:", micro_cm)
    print("Macro_CM:", macro_cm)

    print("Mirco Accuracy:", micro_acc)
    print("Macro Accuracy:", macro_acc)
    print("Accuracy stdev:", stdev_acc)
    
    print("Mirco MCC:", micro_mcc)
    print("Macro MCC:", macro_mcc)
    print("MCC stdev:", stdev_mcc)
    """
    print("Mirco Vio Precision:", vio_micro_prec, "Mirco Non-vio Precision:", non_micro_prec)
    print("Macro Vio Precision:", macro_vio_prec, "Macro Non-vio Precision:", macro_non_prec)
    """
    #print("Mirco Vio Recall:", vio_micro_recall, "Mirco Non-vio Recall:", non_micro_recall)
    print("Macro Vio Recall:", macro_vio_recall, "Macro Non-vio Recall:", macro_non_recall)
    print("Vio Recall stdev:", stdev_vio_recall, "Non-vio Recall stdev:", stdev_non_recall)
    
    print("Productivity (per person):", productivity)
    print("Productivity stdev:", stdev_productivity)

    print('\n')
    return macro_mcc_list, productivity_list


"""
def append_confuse adds the three individual confusion matrix lists to the target group 
confusion matrix lists.
"""
def append_confuse(ind_matrices, target_cm):
    # Should only be three indices: 'total', 'circumstances', 'domestic law' 
    if len(ind_matrices) > 3:
        raise ValueError(f"Unexpected number of confusion matrices: {len(ind_matrix)}")
    for idx_mat, matrix in enumerate(ind_matrices):
        target_cm[idx_mat].append(ind_matrices[idx_mat])                       
    return target_cm


"""
def get_accuracy calculates raw accuracy: confuse[0] = tp, confuse[1] = tn,
confuse[2] = fp, confuse[3] = fn
"""
def get_accuracy(confuse):
    assert len(confuse) == 4
    accuracy = (confuse[0] + confuse[1]) / (sum(confuse)) 
    return accuracy
    

"""
def dataframe_mccs compares filename to agents in loaded excel file
and adds the relevant mcc score to the dataframe.
"""
def dataframe_ext(filename, df, ind_matrices, analysis_type):
    # Extract the identifier from the filename
    identifier_match = re.search(r'^([a-zA-Z]+)[ -]?', filename, re.IGNORECASE)
    if identifier_match:
        identifier = identifier_match.group(1).strip().lower()  # Convert to lowercase and remove spaces
        # Find the matching row in df (case-insensitive comparison)
        if analysis_type == 'quiz_comp':
            matching_row = df[df['Agent name:'].str.lower() == identifier]
        else:
            matching_row = df[df['2. What is your anonymised identifier? Agent:'].str.lower() == identifier]
        if not matching_row.empty:
            mcc_value = get_mcc(ind_matrices[0])  # Replace with actual MCC calculation
            matching_index = matching_row.index[0]  # Get the index of the matching row
            df.at[matching_index, 'Participant MCC'] = mcc_value
            df.at[matching_index, 'Productivity'] = sum(ind_matrices[0])
        elif analysis_type in ['conf_comp', 'kanst_comp']:
            raise ValueError(f"Can't find matching agent in df for filename: {filename}")
    return df
    

"""
def get_mcc calculates MCC score: confuse[0] = tp, confuse[1] = tn,
confuse[2] = fp, confuse[3] = fn
"""
def get_mcc(confuse):
    assert len(confuse) == 4
    numerator = confuse[1] * confuse[0] - confuse[2] * confuse[1]
    denominator = (confuse[0] + confuse[2]) * (confuse[0] + confuse[3]) * (confuse[1] + confuse[2]) * (confuse[1] + confuse[3]) 
    mcc_score = numerator / denominator**0.5
    return mcc_score


"""
def get_precision calculates precision: confuse[0] = tp, confuse[1] = tn,
confuse[2] = fp, confuse[3] = fn
"""
def get_precision(confuse):
    assert len(confuse) == 4
    vio_precision = confuse[0] / (confuse[0] + confuse[2]) 
    non_precision = confuse[1] / (confuse[1] + confuse[3])
    return vio_precision, non_precision
    

"""
def get_recall calculates recall: confuse[0] = tp, confuse[1] = tn,
confuse[2] = fp, confuse[3] = fn
"""
def get_recall(confuse):
    assert len(confuse) == 4
    vio_recall = confuse[0] / (confuse[0] + confuse[3]) 
    non_recall = confuse[1] / (confuse[1] + confuse[2])
    return vio_recall, non_recall


"""
def get_confusion_matrices calculates the three confusion matrix, for a given class and
counter class. The target class is expected to be 'violation', and the counter
class is expected to be 'nonviolation'. The first matrix is over all classifications; the second is
for circumstances; the third matrix is for domestic law.
"""
def get_confusion_matrices(df, filename, target_class = 'violation', counter_class = 'nonviolation'):
    # Get index of row for participant's classifications
    estimate_idx = df[df.iloc[:, 0] == 'Your verdict'].index[0]
    actual_idx = df[df.iloc[:, 0] == 'True Verdict'].index[0]
    desc_idx = df[df.iloc[:, 0] == 'Case file'].index[0]
    
    # Get the column index for the last case evaluated by the study participant      
    last_column_idx = df.iloc[estimate_idx].tolist().index(np.nan) - 1
    column_values = df.iloc[:, last_column_idx]
    
    # Initialise confusion matrix [tp, tn, fp, fn]]
    tot_confuse = np.zeros(4)
    circ_confuse = np.zeros(4)
    dom_confuse = np.zeros(4) 
    total_count = 0
    
    # Iterate across all columns containing participant's classifications
    for column_idx in range(1, last_column_idx + 1):
        temp_confuse = [0, 0, 0, 0]
        if pd.isna(df.iloc[estimate_idx, column_idx]):
            print("Filename:", filename)
            raise ValueError(f"Unexpected nan value in column: {column_idx}")  
        total_count += 1
        if df.iloc[actual_idx, column_idx] == target_class:
            if df.iloc[estimate_idx, column_idx] == target_class:
                temp_confuse[0] = 1
            elif df.iloc[estimate_idx, column_idx] == counter_class:
                temp_confuse[3] = 1
            else:
                print("Filename:", filename)
                raise ValueError(f"Unexpected value for estimated verdict in column: {column_idx}") 
        if df.iloc[actual_idx, column_idx] == counter_class:
            if df.iloc[estimate_idx, column_idx] == target_class:
                temp_confuse[2] = 1
            elif df.iloc[estimate_idx, column_idx] == counter_class:
                temp_confuse[1] = 1
            else:
                print("Filename:", filename)
                raise ValueError(f"Unexpected value for estimated verdict in column: {column_idx}")
        tot_confuse += temp_confuse
        if re.search('Circumstances', df.iloc[desc_idx, column_idx], re.IGNORECASE):
            circ_confuse += temp_confuse
        elif re.search('Domestic', df.iloc[desc_idx, column_idx], re.IGNORECASE):
            dom_confuse += temp_confuse
        else:
            print("Filename:", filename)
            print("output:", df.iloc[desc_idx, column_idx])
            raise ValueError(f"Unexpected value for facts description type in column: {column_idx}") 
            
    if total_count != sum(tot_confuse):
        print("tot_confuse:", tot_confuse)
        print("count:", total_count)
        raise ValueError(f"Expected sum of confusion matrix to be equal to count of instances: {filename}")

    return tot_confuse, circ_confuse, dom_confuse


"""
def main requests user input to indicate what type of analysis is required.
"""
def main():
    dataframes = {}
    
    while True:
        action = input("Please enter an action (options: 'load', 'plot', 'test', 'exit'): ").strip().lower()
        
        if action == 'load':
            file_id = input("Please enter the file_id (options: 'zero_shot', 'few_shot', 'full'): ").strip()
            if file_id not in ["zero_shot", "few_shot", "full"]:
                print("Invalid file_id input.")
                continue

            analysis_type = input("Please enter the analysis_type (options: 'ind_scores', 'group_scores', 'model_anlys', 'domain_anlys', 'conf_comp', 'circ_dome', 'kanst_comp', 'quiz_comp'): ").strip()
            if analysis_type not in ['ind_scores', 'group_scores', 'model_anlys', 'domain_anlys', 'conf_comp', 'circ_dome', 'kanst_comp', 'quiz_comp']:
                print("Invalid analysis_type input.")
                continue
            
            # Calling read_individual to load main analysis           
            if file_id == 'zero_shot':
                dataframes[file_id] = read_individual(2, analysis_type)
            elif file_id == 'full':
                dataframes[file_id] = read_individual(4, analysis_type)
            elif file_id == 'few_shot':
                dataframes[file_id] = read_individual(file_id, analysis_type)
            else:
                print("Invalid file_id input.")
                continue

        elif action == 'plot':
            if not dataframes:
                print("No data loaded. Load data first using the 'load' action.")
                continue

            file_id = input(f"Please enter the file_id for plotting (loaded options: {list(dataframes.keys())}): ")
            x_axis = 'Productivity'
            y_axis = 'Participant MCC'
            
            # Try converting file_ids to integers if they represent integers
            try:
                file_id = int(file_id)
            except ValueError:
                pass         
            if file_id not in dataframes:
                print(f"Invalid file_id input: {file_id}")
                continue
            
            print("Getting ready to plot...")
            if file_id == 'full':
                plot_scatter(dataframes[file_id][x_axis], dataframes[file_id][y_axis], [0, 200], [-0.3, 0.6], str(file_id), x_axis, y_axis)
            else:
                plot_scatter(dataframes[file_id][x_axis], dataframes[file_id][y_axis], [0, 120], [-0.3, 0.6], str(file_id), x_axis, y_axis)            

        elif action == 'test':
            if not dataframes:
                print("No data loaded. Load data first using the 'load' action.")
                continue

            file_id_1 = input(f"Please enter the first file_id for testing (loaded options: {list(dataframes.keys())}): ")
            data_1 = input("Please enter the data_type for the first file (options: 'Productivity', 'Participant MCC'): ").strip()
            file_id_2 = input(f"Please enter the second file_id for testing (loaded options: {list(dataframes.keys())}): ")
            data_2 = input("Please enter the data_type for the second file (options: 'Productivity', 'Participant MCC'): ").strip()
            
            # Try converting file_ids to integers if they represent integers
            try:
                file_id_1 = int(file_id_1)
            except ValueError:
                pass   
            try:
                file_id_2 = int(file_id_2)
            except ValueError:
                pass  
            if file_id_1 not in dataframes or file_id_2 not in dataframes:
                print("Invalid file_id input(s).")
                continue
            
            print("Testing correlation between", file_id_1, data_1, "and", file_id_2, data_2)
            stats_tests(dataframes[file_id_1][data_1], dataframes[file_id_2][data_2])
        
        elif action == 'exit':
            break
        
        else:
            print("Invalid action input.")
            
    return
    

"""
def plot_scatter plots a scatter plot of columns from two dataframes.
Expected parameters: df1[col], df2[col].
"""
def plot_scatter(data1, data2, limits1, limits2, data_type, label1, label2):
    ## Setting write directory
    write_path = 'Analysis'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
        
    plt.figure(figsize=(8, 6))
    plt.scatter(data1, data2, alpha=0.5)
    plt.xlabel(label1)
    plt.ylabel(label2)
    
    # Set the limits of the x-axis and y-axis
    plt.xlim(limits1)
    plt.ylim(limits2)
    
    plt.grid(True)
    plot_name = data_type + '_' + label1 +   '_' + label2 + '.png'
    plt.savefig(os.path.join(write_path, plot_name))
    plt.show()
    plt.close()
    return


"""
def print_anlys prints the analysis output.
"""
def print_anlys(title_names, save_lists):
    assert len(title_names) == len(save_lists)
    mcc_lists = []
    product_lists = []
    for idx, item in enumerate(title_names):
        print(title_names[idx])
        new_mcc, new_product = analyse_cms(save_lists[idx][0])
        mcc_lists.append(new_mcc)
        product_lists.append(new_product)
    return mcc_lists, product_lists


"""
def read_individual takes k as input, indicating the session of the research study
from which the data is to be read (k=4 by default). Outputs processed excel files
to standardise the participants classifications, and adds the true verdicts.
analysis_type can take values in ['ind_scores', 'group_scores', 'model_anlys', 'domain_anlys', 
'conf_comp', 'circ_dome', 'kanst_comp', 'quiz_comp'].
"""
def read_individual(k=4, analysis_type=0):

    # Common path elements for all files
    raw_path = os.path.join('Analysis', 'Processed')   
    # Define the pattern to match the file name of the session (should be k=4 for final session)
    pattern = fr'session.*{k}.*\.xlsx' 
    print("Session:", k, '\n')
    all_groups = [[],[],[]]
    # Define lists for collecting model groups and for non-model groups
    no_model = [[],[],[]]
    with_model = [[],[],[]]     
    # Define lists for collecting cs, domain, and law groups
    weak = [[],[],[]]
    moderate = [[],[],[]]
    strong = [[],[],[]]
    # Loading survey reponses dataframe
    survey_df = pd.read_excel('survey_responses.xlsx')
    quiz_df = pd.read_excel('model_2nd_quiz_responses.xlsx')
    if analysis_type == 'conf_comp':
        # Mapping dictionary for converting text to numeric values
        confidence_mapping = {'Very inaccurately': 0, 'Slightly inaccurately': 1, 'Roughly balanced': 2, 'Slightly accurately': 3, 'Very accurately': 4}
        # Apply the mapping to each column and sum the results
        survey_df['Reported Confidence'] = (survey_df['18.1. Immediately after training'].map(confidence_mapping) + survey_df['18.2. At the end of the project'].map(confidence_mapping))
    elif analysis_type == 'kanst_comp':
        location_answers = {'Brussels': 0, 'Luxembourg': 0, 'the Hague': 0, 'Strasbourg': 1}      
        polity_answers = {True: 0, False: 1}
        # Apply the mapping to each column and sum the results
        survey_df['Kanstatsin Knowledge'] = (survey_df['4. Where is the European Court of Human Rights located?'].map(location_answers) + survey_df['5. The European Court of Human Rights is the tribunal created by the United Nations.'].map(polity_answers) + survey_df['6. The European Court of Human Rights is the tribunal created by the European Union.'].map(polity_answers))
    # Iterating through all session outputs  
    for root, dirs, files in os.walk(raw_path):
        for group_path in dirs:
            print("GROUP PATH:", group_path, '\n')
            subdirectory_path = os.path.join(root, group_path)        
            # Initialise group confusion matrix
            group_conf_mats = [[],[],[]]           
            # Iterate over the files in the subdirectory
            for filename in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, filename)      
                # Check if the file is an Excel file and matches the pattern
                if filename.endswith(".xlsx") and re.search(pattern, filename, re.IGNORECASE):                   
                    # Read the Excel file using pandas
                    df = pd.read_excel(file_path)
                    # Perform individual analysis
                    ind_matrices = get_confusion_matrices(df, filename, 'violation', 'nonviolation')
                    survey_df = dataframe_ext(filename, survey_df, ind_matrices, analysis_type)
                    if analysis_type == 'quiz_comp':
                        quiz_df = dataframe_ext(filename, quiz_df, ind_matrices, analysis_type)
                    all_groups = append_confuse(ind_matrices, all_groups)
                    if analysis_type == 'ind_scores':
                        print("Filename:", filename)
                        ind_acc = get_accuracy(ind_matrices[0])
                        ind_mcc = get_mcc(ind_matrices[0])
                        print("Accuracy:", ind_acc)
                        print("MCC:", ind_mcc, '\n')                   
                    elif analysis_type == 'group_scores':
                        group_conf_mats = append_confuse(ind_matrices, group_conf_mats)
                    elif analysis_type == 'model_anlys': 
                        no_model, with_model = splice_analysis(['Alpha', 'Omega'], [no_model, with_model], group_path, ind_matrices)
                    elif analysis_type == 'domain_anlys':
                        weak, moderate, strong = splice_analysis(['CS', 'Law', 'Domain'], [weak, moderate, strong], group_path, ind_matrices)          
            if analysis_type == 'group_scores':          
                print("Overall analysis:")
                analyse_cms(group_conf_mats[0])
    print('ALL GROUPS')            
    analyse_cms(all_groups[0])
    if analysis_type == 'circ_dome':
        print('CIRCUMSTANCES OF THE CASE:')
        circ_mcc, circ_prod = analyse_cms(all_groups[1])
        print('RELEVANT LEGAL FRAMEWORK:')
        dome_mcc, dome_prod = analyse_cms(all_groups[2])
        print("Testing correlation between CIRC... and RELE...", '\n')
        stats_tests(circ_mcc, dome_mcc)
    elif analysis_type == 'conf_comp':
        print("Testing correlation between REPORTED CONFIDENCE and PARTICIPANT MCC SCORE", '\n')
        stats_tests(survey_df['Reported Confidence'], survey_df['Participant MCC'])
    elif analysis_type == 'kanst_comp':
        print("Testing correlation between KANSTANT SCORE and PARTICIPANT MCC SCORE", '\n')
        stats_tests(survey_df['Kanstatsin Knowledge'], survey_df['Participant MCC'])
    elif analysis_type == 'quiz_comp':      
        #print(quiz_df[['Agent name:', 'Score', 'Participant MCC']])
        print("Testing correlation between QUIZ SCORE and PARTICIPANT MCC SCORE", '\n')
        stats_tests(quiz_df['Score'], quiz_df['Participant MCC'])
    elif analysis_type == 'model_anlys':   
        mcc_lists, product_lists = print_anlys(["NO MODEL GROUPS:", "WITH MODEL GROUPS:"], [no_model, with_model])
        print("MCC stats:")
        u_statistic, mann_p_value = stats.mannwhitneyu(mcc_lists[0], mcc_lists[1], alternative='two-sided')
        print(f"Mann-Whitney U statistic: {u_statistic}")
        print(f"P-value: {mann_p_value}", '\n')
        print("Productivity stats:")
        u_statistic, mann_p_value = stats.mannwhitneyu(product_lists[0], product_lists[1], alternative='two-sided')
        print(f"Mann-Whitney U statistic: {u_statistic}")
        print(f"P-value: {mann_p_value}")
    elif analysis_type == 'domain_anlys':     
        print_anlys(["WEAK GROUPS:", "MODERATE GROUPS:", "STRONG GROUPS:"], [weak, moderate, strong])          
    return survey_df


"""
def splice_analysis looks up the analysis type names in the files and extracts the relevant
performance data for the overall, circumstances, and relevant domestic law classifications.
"""
def splice_analysis(lookup_names, save_lists, group_path, ind_matrices):
    assert len(lookup_names) == len(save_lists)
    for idx, item in enumerate(lookup_names):
        if re.search(lookup_names[idx], group_path, re.IGNORECASE):
            save_lists[idx] = append_confuse(ind_matrices, save_lists[idx])
    return save_lists
    

"""
def stats_tests performs Spearman's rank correlation test and Mann-Whitney U test
to compare participant performance in accordance with some independent variable.
"""
def stats_tests(list1, list2):

    # Remove NaN values from both lists
    list1 = np.array(list1)
    list2 = np.array(list2)
    list1 = list1[~np.isnan(list1)]
    list2 = list2[~np.isnan(list2)]
    
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


if __name__ == "__main__":
    main()


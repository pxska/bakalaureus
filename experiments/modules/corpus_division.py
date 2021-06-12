import os
import json
import random

# Return a list of all filenames that are in the corpus
def get_filenames(vallakohtufailid_location, directories):
    filenames = []
    for directory in directories:
        files_in_directory = os.listdir(os.path.join(vallakohtufailid_location, directory))
        for file in files_in_directory:
            if file.endswith('.txt'):
                filenames.append(file.replace('.txt', '.json'))
    return filenames

# Get all names of files with hand-tagged tags from the corpus
# Input: a file with id-s of hand-tagged protocols on each line
# Output: a list of filenames that are hand-tagged (should they be in the corpus)
def get_hand_tagged(hand_tagged_protocols_file, filenames):
    protocols_tagged_by_hand_id = []
    protocols_tagged_by_hand = []
    
    with open(hand_tagged_protocols_file, 'r', encoding="UTF-8") as in_f:
        lines = in_f.readlines()
    
    hand_tagged_id = [file_id.strip() for file_id in lines if file_id.strip() != '---']

    hand_tagged_protocols = []
    
    for i in hand_tagged_id:
        res = [x for x in filenames if f'_id{i}_' in x]
        if len(res) == 1:
            hand_tagged_protocols.append(res[0])

    return hand_tagged_protocols

# Get the distribution (percentage) of all labels across the corpus
def get_labels_distribution(filenames, gold_standard_files):
    all_annotations = []
    ideal_distribution = dict()
    
    for file in filenames:
        with open(os.path.join(gold_standard_files, file), 'r', encoding="UTF-8)") as f:
            data = json.load(f)
        for dictionary in data.get('layers')[0].get('spans'):
            all_annotations.append(dictionary.get('annotations')[0].get('nertag'))
            
    unique_labels = set(all_annotations)
    
    for label in unique_labels:
        percentage = (all_annotations.count(label) / len(all_annotations)) * 100
        ideal_distribution[label] = percentage
    
    return ideal_distribution

# Get the labels (and the number of those labels) of every document in the corpus
def get_documents_ne_statistics(filenames, gold_standard_files):
    statistics = dict()
    for file in filenames:
        ner_annotations = []
        
        with open(os.path.join(gold_standard_files, file), 'r', encoding="UTF-8)") as in_f:
            data = json.load(in_f)
            
        for dictionary in data.get('layers')[0].get('spans'):
            ner_annotations.append(dictionary.get('annotations')[0].get('nertag'))

        statistics_for_file = dict()
        
        for annotation in set(ner_annotations):
            statistics_for_file[str(annotation)] = ner_annotations.count(annotation)
            
        statistics[file] = statistics_for_file

    return statistics

# Swap two items in two lists and return new lists
def swap_items_in_lists(A, B, i, j):
    temp_b = B[j]
    B[j] = A[i]
    A[i] = temp_b
    return A, B

# Calculate proportions (percentages) of labels in a selected number of protocols
def calculate_proportions(filenames, statistics):
    statistics_for_proportion = {}
    
    for file in filenames:
        statistics_for_proportion[file] = statistics[file]

    all_annotations = list()
    for item in statistics_for_proportion.values():
        for key in item:
            for i in range(item[key] + 1):
                all_annotations.append(key)
                
    proportions = dict()
    for file in statistics_for_proportion:
        for key in statistics_for_proportion[file].keys():
            proportion = all_annotations.count(key) / len(all_annotations) * 100
            proportions[key] = proportion
        
    return proportions

# Reduce scores in the two largest lists (lower score is better)
# The score is calculated based on the deviation from the ideal distribution
def improve_scores(largest, second_largest, statistics):
    score_largest_old = calculate_score(largest, statistics, ideal_distribution)
    score_second_largest_old = calculate_score(second_largest, statistics, ideal_distribution)
    for i in range(len(largest)):
        for j in range(len(second_largest)):
            largest, second_largest = swap_items_in_lists(largest, second_largest, i, j)
            
            score_largest_new = calculate_score(largest, statistics, ideal_distribution)
            score_second_largest_new = calculate_score(second_largest, statistics, ideal_distribution)

            if score_largest_old > score_largest_new and score_second_largest_old > score_second_largest_new:
                print(f"(!) {i, j}: the score was better in both lists")
                score_largest_old = score_largest_new
                score_second_largest_old = score_second_largest_new
                i += 1
            elif score_largest_old < score_largest_new and score_second_largest_old > score_second_largest_new:
                print(f"(!) {i, j}: the score was better only in the first lists")
                swap_items_in_lists(largest, second_largest, i, j)
                score_largest_old = score_largest_old
                score_second_largest_old = score_second_largest_old
            elif score_largest_old > score_largest_new and score_second_largest_old < score_second_largest_new:
                print(f"(!) {i, j}: the score was better only in the second lists")
                swap_items_in_lists(largest, second_largest, i, j)
                score_largest_old = score_largest_old
                score_second_largest_old = score_second_largest_old
            else:
                print(f"(!) {i, j}: the score was better in neither of the lists")
                swap_items_in_lists(largest, second_largest, i, j)
                score_largest_old = score_largest_old
                score_second_largest_old = score_second_largest_old
    return largest, second_largest

# Remove hand-tagged files from a list
def remove_hand_tagged(least_hand_tagged, other_files, statistics, ideal_distribution, protocols_tagged_by_hand):
    for i, x in enumerate(set(least_hand_tagged).intersection(protocols_tagged_by_hand)):
        print(i+1, x)
        index = least_hand_tagged.index(x)
        score_old = calculate_score(least_hand_tagged, statistics, ideal_distribution)

        for files in other_files:
            breaking = "no"
            score_old_output = calculate_score(files, statistics, ideal_distribution)

            for j, y in enumerate(files):
                if y in protocols_tagged_by_hand:
                    continue
                else:
                    swap_items_in_lists(least_hand_tagged, files, index, j)
                    score_new = calculate_score(least_hand_tagged, statistics, ideal_distribution)
                    score_new_output = calculate_score(files, statistics, ideal_distribution)
                    if score_new < score_old and score_new_output < score_old_output:
                        print(f'(!) Changed out file {y}')
                        breaking = "yes"
                        break
                    elif score_new == score_old and score_new_output == score_old_output or \
                         abs(score_new - score_old) < 0.05 and abs(score_new_output-score_old_output) < 0.05:
                        print(f'(!) Changed out file {y}')
                        breaking = "yes"
                        break
                    else:
                        swap_items_in_lists(least_hand_tagged, files, index, j)
                        continue
            if breaking == "yes":
                break
    return least_hand_tagged, other_files

# Calculate score based on the deviation from the ideal distribution
def calculate_score(filenames, statistics, ideal_distribution):
    proportions = calculate_proportions(filenames, statistics)
    score = 0
    for proportion in proportions:
        ideal_distribution_proportion = ideal_distribution[proportion]
        current_proportion = proportions[proportion]
        
        if current_proportion == ideal_distribution_proportion:
            score += 0
        else:
            score += abs(ideal_distribution_proportion - current_proportion)
    
    return score

# Divide files into n even parts
def n_even_chunks(filenames, n):
    files = []
    last = 0
    for i in range(1, n+1):
        current = int(round(i* (len(filenames) / n)))
        files.append(filenames[last:current])
        last = current
    return files

# Generate random division of the files into n even parts
def generate_random_division(filenames, n):
    random_distributions = []
    for i in range(n):
        filenames = random.sample(filenames, len(filenames))
        random_distributions.append(filenames)
    
    return random_distributions
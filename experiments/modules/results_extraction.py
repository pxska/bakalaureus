import pandas as pd
import os
import json

from estnltk.converters import json_to_text
from nervaluate import Evaluator

def extract_annotations(no_goldstandard_annotations, trained_files_location, testing_files_location, file):
    gold = list()
    test = list()
    
    if file.endswith('.json') and file not in no_goldstandard_annotations:
        with open(os.path.join(trained_files_location, file), 'r', encoding='UTF-8') as f_test, \
             open(os.path.join(testing_files_location, file), 'r', encoding='UTF-8') as f_gold:
                test_import = json_to_text(f_test.read())
                gold_import = json_to_text(f_gold.read())

        for i in range(len(gold_import['gold_ner'])):
            ner = gold_import['gold_ner'][i]
            label = ner.nertag
            start = int(ner.start)
            end = int(ner.end)
            gold.append({"label": label, "start": start, "end": end})

        for i in range(len(test_import['flat_ner'])):
            ner = test_import['flat_ner'][i]
            label = ner.nertag[0]
            start = int(ner.start)
            end = int(ner.end)
            test.append({"label": label, "start": start, "end": end})
    
    return gold, test

def extract_results(model_dir, files, no_goldstandard_annotations, trained_files_location, testing_files_location, results_location):
    
    if (len(set(files.values())) == 1):
        gold_ner = list()
        test_ner = list()
        
        for file in [key for key, value in files.items()]:
            gold, test = extract_annotations(no_goldstandard_annotations, trained_files_location, testing_files_location, file)
            results_gold += gold
            results_test += test
        
        evaluator = Evaluator(gold_ner, test_ner, tags=['ORG', 'PER', 'MISC', 'LOC', 'LOC_ORG'])
        results, results_per_tag = evaluator.evaluate()
        all_results = (results, results_per_tag)
        
    else:
        all_results = {}
        
        for subdistribution in sorted(set(files.values())):
            gold_ner = list()
            test_ner = list()
            
            for file in [key for key, value in files.items() if int(value) == int(subdistribution)]:
                gold, test = extract_annotations(no_goldstandard_annotations, trained_files_location, testing_files_location, file)
                gold_ner.append(gold)
                test_ner.append(test)
    
            evaluator = Evaluator(gold_ner, test_ner, tags=['ORG', 'PER', 'MISC', 'LOC', 'LOC_ORG'])
            results, results_per_tag = evaluator.evaluate()
            all_results[subdistribution] = (results, results_per_tag)
        
    with open(os.path.join(results_location, 'results.txt'), 'w+') as out_f:
        out_f.write(json.dumps(all_results))
    
    print(f'Results have been saved to {results_location}/results.txt')

def results_by_subdistribution(results_json, files):
    data = dict()
    
    if len(set(files.values())) == 1:
        correct = results_json[0]['strict']['correct']
        actual = results_json[0]['strict']['actual']
        possible = results_json[0]['strict']['possible']
        
        precision = (correct / actual)
        recall = (correct / possible)
        f1 = 2 * ((precision * recall) / (precision + recall))
        
        data = {'Precision': precision, 'Recall': recall, 'F1': f1}
        results_df = pd.DataFrame(data, index=[0])
        
    else:
        correct_all = 0
        actual_all = 0
        possible_all = 0
        for subdistribution in sorted(set(files.values())):    
            correct = results_json[subdistribution][0]['strict']['correct']
            actual = results_json[subdistribution][0]['strict']['actual']
            possible = results_json[subdistribution][0]['strict']['possible']

            correct_all += correct
            actual_all += actual
            possible_all += possible
        
            precision = (correct / actual)
            recall = (correct / possible)
            f1 = 2 * ((precision * recall) / (precision + recall))

            data[str(subdistribution)] = {
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
    
        precision = correct_all / actual_all
        recall = correct_all / possible_all
        f1 = 2 * ((precision * recall) / (precision + recall))
    
        data["Total"] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
        results_df = pd.DataFrame(data)
        results_df.columns.name = "Alamhulk"
    
    return results_df

def results_by_named_entity(results_json, files):
    data = dict()
    totals = dict()

    for subdistribution in sorted(set(files.values())):
        by_kind = dict()
        
        for key in list(results_json[subdistribution][1].keys()):
            correct_all = 0
            actual_all = 0
            possible_all = 0
            
            correct = results_json[subdistribution][1][str(key)]['strict']['correct']
            correct_all += correct
            actual = results_json[subdistribution][1][str(key)]['strict']['actual']
            actual_all += actual
            possible = results_json[subdistribution][1][str(key)]['strict']['possible']
            possible_all += possible
            
            precision = (correct / actual)
            recall = (correct / possible)
            f1 = 2 * ((precision * recall) / (precision + recall))

            by_kind[str(key) + "_precision"] = precision
            by_kind[str(key) + "_recall"] = recall
            by_kind[str(key) + "_f1score"] = f1
        
        data[str(subdistribution)] = by_kind

    for key, value in data.items():
        for name, score in value.items():
            if name in totals:
                totals[name] = (totals.get(name) + score)
            else:
                totals[name] = score
    
    for key, value in totals.items():
        totals[key] = value / int(len(set(files.values())))
    
    data['Total'] = totals
    
    results_df = pd.DataFrame(data)
    
    return results_df
    
def confusion_matrix(model_dir, files):
    gold_ner = []
    test_ner = []

    for file in files:
        gold, test = extract_annotations(no_goldstandard_annotations, trained_files_location, testing_files_location, file)
        
        gold_ner.append(gold)
        test_ner.append(test)
    
    uus_gold_ner = []
    uus_test_ner = []
    
    for i in range(len(gold_ner)):
        for j in range(len(test_ner[i])):
            element_test = test_ner[i][j]
            for element_gold in gold_ner[i]:
                if element_test['start'] == element_gold['start'] and element_test['end'] == element_gold['end']:
                    uus_gold_ner.append(element_gold)
                    uus_test_ner.append(element_test)
                    
    y_true = pd.Series([x['label'] for x in uus_gold_ner], name="Actual")
    y_pred = pd.Series([x['label'] for x in uus_test_ner], name="Predicted")
    return y_true, y_pred
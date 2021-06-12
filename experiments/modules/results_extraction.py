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

def results_by_subdistribution(results_json):
    correct_all = 0
    actual_all = 0
    possible_all = 0
    df = dict()

    for i in ['1', '2', '3', '4', '5']:    
        correct = results_json[i][0]['strict']['correct']
        correct_all += correct
        
        actual = results_json[i][0]['strict']['actual']
        actual_all += actual
        
        possible = results_json[i][0]['strict']['possible']
        possible_all += possible
        
        precision = (correct / actual)
        recall = (correct / possible)
        f1 = 2 * ((precision * recall) / (precision + recall))
        
        df[str(i)] = [precision, recall, f1]
    
    precision = correct_all / actual_all
    recall = correct_all / possible_all
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    df["Total"] = [precision, recall, f1]
    
    dataframe = pd.DataFrame(df, index=["Precision", "Recall", "F1-score"])
    dataframe.columns.name = "Alamhulk"
    
    return dataframe

def results_by_named_entity(results_json):
    df = dict()
    totals = dict()

    for i in ['1', '2', '3', '4', '5']:
        train = []
        by_kind = dict()
        for j in ['1', '2', '3', '4', '5']:
            if j == i:
                subdistribution_for_testing = j
            else:
                train.append(j)

        for key in list(results_json[i][1].keys()):
            correct_all = 0
            actual_all = 0
            possible_all = 0
            correct = results_json[i][1][str(key)]['strict']['correct']
            correct_all += correct
            actual = results_json[i][1][str(key)]['strict']['actual']
            actual_all += actual
            possible = results_json[i][1][str(key)]['strict']['possible']
            possible_all += possible

            precision = (correct / actual)
            recall = (correct / possible)
            f1 = 2 * ((precision * recall) / (precision + recall))

            precisionname = str(key) + "_precision"
            recallname = str(key) + "_recall"
            f1scorename = str(key) + "_f1score"

            by_kind[precisionname] = precision
            by_kind[recallname] = recall
            by_kind[f1scorename] = f1

        df[str(subdistribution_for_testing)] = by_kind

    for key, value in df.items():
        for name, score in value.items():
            if name in totals:
                totals[name] = (totals.get(name) + score)
            else:
                totals[name] = score

    for key, value in totals.items():
        totals[key] = value/5

    df["Total"] = totals
    return df
    
def confusion_matrix(model_dir, files):
    gold_ner = []
    test_ner = []

    for file in files:
        appendable_gold_ner = []
        appendable_test_ner = []
        
        with open(os.path.join('models', model_dir, 'vallakohtufailid-trained-nertagger', file), 'r', encoding='UTF-8') as f_test, \
             open(os.path.join('..', 'data', 'vallakohtufailid-json-flattened', file), 'r', encoding='UTF-8') as f_gold:
                test_import = json_to_text(f_test.read())
                gold_import = json_to_text(f_gold.read())

                for i in range(len(gold_import['gold_ner'])):
                    ner = gold_import['gold_ner'][i]
                    label = ner.nertag
                    start = int(ner.start)
                    end = int(ner.end)
                    appendable_gold_ner.append({"label": label, "start": start, "end": end})

                for i in range(len(test_import['flat_ner'])):
                    ner = test_import['flat_ner'][i]
                    label = ner.nertag[0]
                    start = int(ner.start)
                    end = int(ner.end)
                    appendable_test_ner.append({"label": label, "start": start, "end": end})

        gold_ner.append(appendable_gold_ner)
        test_ner.append(appendable_test_ner)
    
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
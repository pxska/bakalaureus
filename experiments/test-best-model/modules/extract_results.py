from estnltk.converters import json_to_text
from nervaluate import Evaluator
import sklearn_crfsuite
import nereval
import pandas as pd
import os
import json
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics

def extract_results_to_txt_file(model_dir, files):
    gold_ner = []
    test_ner = []

    for file in files:
        appendable_gold_ner = []
        appendable_test_ner = []
        
        with open(os.path.join('models', model_dir, 'vallakohtufailid-trained-nertagger', file), 'r', encoding='UTF-8') as f_test, \
             open(os.path.join('test', 'flattened_json_files', file), 'r', encoding='UTF-8') as f_gold:
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
    print(gold_ner, test_ner)
    
    evaluator = Evaluator(gold_ner, test_ner, tags=['ORG', 'PER', 'MISC', 'LOC', 'LOC_ORG'])
    results, results_per_tag = evaluator.evaluate()
    all_results = (results, results_per_tag)
    
    print("(!) Tulemused arvutatud")
    
    with open(os.path.join('models', model_dir, 'results.txt'), 'w+') as results_file:
        results_file.write(json.dumps(all_results))
    print(f'(!) Tulemused kirjutatud faili {results_file}')
    
    return all_results

def display_results_by_subdistribution(results_json):
    df = dict()
      
    correct = results_json[0]['strict']['correct']
    actual = results_json[0]['strict']['actual']
    possible = results_json[0]['strict']['possible']
    
    precision = correct / actual
    recall = correct / possible
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    df["Total"] = [precision, recall, f1]
    
    dataframe = pd.DataFrame(df, index=["Precision", "Recall", "F1-score"])
    
    return dataframe

def display_results_by_named_entity(results_json):
    by_kind = dict()
    df = dict()
    
    for key in list(results_json[1].keys()):
        
        correct = results_json[1][str(key)]['strict']['correct']
        actual = results_json[1][str(key)]['strict']['actual']
        possible = results_json[1][str(key)]['strict']['possible']

        precision = (correct / actual)
        recall = (correct / possible)
        
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * ((precision * recall) / (precision + recall))

        precisionname = str(key) + "_precision"
        recallname = str(key) + "_recall"
        f1scorename = str(key) + "_f1score"

        by_kind[precisionname] = precision
        by_kind[recallname] = recall
        by_kind[f1scorename] = f1

        df['results'] = by_kind

    return df
    
def display_confusion_matrix(model_dir, files):
    gold_ner = []
    test_ner = []

    for file in files:
        appendable_gold_ner = []
        appendable_test_ner = []
        
        with open(os.path.join('models', model_dir, 'vallakohtufailid-trained-nertagger', file), 'r', encoding='UTF-8') as f_test, \
             open(os.path.join('test', 'flattened_json_files', file), 'r', encoding='UTF-8') as f_gold:
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
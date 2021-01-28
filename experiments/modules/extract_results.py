from estnltk.converters import json_to_text
from nervaluate import Evaluator
import sklearn_crfsuite
import nereval
import pandas as pd
import os
import json
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics

files_not_working = ['J2rva_Tyri_V22tsa_id22177_1911a.json', \
                     'J2rva_Tyri_V22tsa_id18538_1894a.json', \
                     'J2rva_Tyri_V22tsa_id22155_1911a.json', \
                     'Saare_Kihelkonna_Kotlandi_id18845_1865a.json', \
                     'P2rnu_Halliste_Abja_id257_1844a.json', \
                     'Saare_Kaarma_Loona_id7575_1899a.json', \
                     'J2rva_Tyri_V22tsa_id22266_1913a.json', \
                     'J2rva_Tyri_V22tsa_id22178_1912a.json']

def extract_results_to_txt_file(model_dir, files):
    all_results = {}
    
    for subdistribution in [1, 2, 3, 4, 5]:
        gold_ner = []
        test_ner = []
        training_subdistributions = []
        for y in [1, 2, 3, 4, 5]:
            if y == subdistribution:
                subdistribution_for_testing = y
            else:
                training_subdistributions.append(y)

        for file in {key: value for key, value in files.items() if int(value) == subdistribution_for_testing}:
            appendable_gold_ner = []
            appendable_test_ner = []

            if file.endswith(".json"):
                if file in files_not_working:
                    continue
                else:
                    with open(os.path.join('models', model_dir, 'vallakohtufailid-trained-nertagger', file), 'r', encoding='UTF-8') as f_test, \
                        open(os.path.join('..', 'data', 'vallakohtufailid-json-flattened', file), 'r', encoding='UTF-8') as f_gold:
                            test_import = json_to_text(f_test.read())
                            gold_import = json_to_text(f_gold.read())

                            # The commented part is needed for word-level-ner.
                            '''
                            for i in range(len(gold_import['flat_gold_wordner'])):
                                tag = gold_import['flat_gold_wordner'][i].nertag[0]
                                gold.append(tag)
                            for i in range(len(test_import['flat_wordner'])):
                                tag = test_import['flat_wordner'][i].nertag[0]
                                test.append(tag)
                            '''

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
        print(len(gold_ner), len(test_ner))
        evaluator = Evaluator(gold_ner, test_ner, tags=['ORG', 'PER', 'MISC', 'LOC', 'LOC_ORG'])
        results, results_per_tag = evaluator.evaluate()
        all_results[subdistribution_for_testing] = (results, results_per_tag)
    print("Tulemuste ammutamine on lõpetatud.")
    
    with open(os.path.join('models', model_dir, 'results.txt'), 'w+') as results_file:
        results_file.write(json.dumps(all_results))
    
    return all_results

def display_results_by_subdistribution(results_json): 
    correct_all = 0
    actual_all = 0
    possible_all = 0
    df = dict()

    for i in ['1', '2', '3', '4', '5']:
        train = []
        for j in ['1', '2', '3', '4', '5']:
            if j == i:
                subdistribution_for_testing = j
            else:
                train.append(j)
        correct = results_json[i][0]['strict']['correct']
        correct_all += correct
        actual = results_json[i][0]['strict']['actual']
        actual_all += actual
        possible = results_json[i][0]['strict']['possible']
        possible_all += possible
        precision = (correct / actual)
        recall = (correct / possible)
        f1 = 2 * ((precision * recall) / (precision + recall))
        df[str(subdistribution_for_testing)] = [precision, recall, f1]

    precision = correct_all / actual_all
    recall = correct_all / possible_all
    f1 = 2 * ((precision * recall) / (precision + recall))
    df["Total"] = [precision, recall, f1]
    dataframe = pd.DataFrame(df, index=["Precision", "Recall", "F1-score"])
    dataframe.columns.name = "Alamhulk"
    return dataframe

def display_results_by_named_entity(results_json):
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
    
def display_confusion_matrix(gold_ner, test_ner):
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
    
    pd.crosstab(y_true, y_pred)
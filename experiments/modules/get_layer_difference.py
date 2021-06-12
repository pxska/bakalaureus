import re

def find_difference(file, brat_annotations, text_goldner):
    set1 = [re.sub(r'\s\s+', r' ', element[3].strip()) for element in brat_annotations]
    set2 = [re.sub(r'\s\s+', r' ', element.enclosing_text.strip()) for element in text_goldner]
    diff = list(list(set(set1)-set(set2)) + list(set(set2)-set(set1)))
    if (diff):
        print(f"(!) Difference in {file} correct annotations and output annotations: {diff}")
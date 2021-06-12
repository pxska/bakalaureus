import re

# If the annotation contains newline character, then indexes will contain ';' at the linebreak (e.g. 388 393;394 398 )
def collect_annotations( in_f ):
    indexes_on_line_split = re.compile(r' (\d+) (\d+;\d+ ){1,}(\d+)$')
    annotations = []
    split_lines_ahead = 0
    for line in in_f:
        line = line.rstrip('\n')
        items = line.split('\t')
        if split_lines_ahead > 0:
            split_lines_ahead -= 1
            last_item = annotations[-1]
            new_tuple = (last_item[0],last_item[1],last_item[2],(last_item[3]+line),last_item[4])
            annotations[-1] = new_tuple
            continue
        if len(items) == 3:
            indexes_str = items[1]
            if indexes_str.count(';') > 0:
                split_lines_ahead += indexes_str.count(';')
            indexes_str = indexes_on_line_split.sub(' \\1 \\3', indexes_str)
            tag, start, end = indexes_str.split()
            annotations.append( (tag, start, end, items[2], items[0]) )
    seen = set()
    removed_duplicates_annotations = []
    for a, b, c, d, e in annotations:
        if not b in seen:
            seen.add(b)
            removed_duplicates_annotations.append((a, b, c, d, e))
        else:
            for index, item in enumerate(removed_duplicates_annotations):
                if item[1] == b and item[2] > c:
                    tuple_without_n = (a, b, c, d, e)
                    item = tuple_without_n
                    removed_duplicates_annotations[index] = item
                elif item[1] == b and item[2] < c:
                    tuple_without_n = (a, b, item[2], d, e)
                    item = tuple_without_n
                    removed_duplicates_annotations[index] = item
                else:
                    continue
    
    for index, item in enumerate(removed_duplicates_annotations):
        if "\xa0" in item[3]:
            replaced = re.sub(r'\s\s+', r' ', item[3].replace(u'\xa0', u' '))
            removed_duplicates_annotations[index] = ( item[0], item[1], item[2], replaced, item[4] )
    
    annotations = sorted(list(set(removed_duplicates_annotations)), key=lambda x: int(x[1]))
    return annotations
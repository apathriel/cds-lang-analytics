import pandas as pd
import spacy
import os
import sys

#probably gonna nest for loops for iterating all the files in subfolders

nlp = spacy.load("en_core_web_md")

input_folder_file_path = os.path.join("data", "input")

input_file_for_testing = os.path.join(input_folder_file_path, "a1", "0100.a1.txt")

input_folders = os.listdir(input_folder_file_path)

with open (input_file_for_testing, "r", encoding="utf-8") as file:
    text = file.read()

doc = nlp(text)

""" for token in doc:
    print(token.i, token.text, token.pos_) """

""" for ent in doc.ents:
    print(ent.text, ent.label_) """

entities = []

for ent in doc.ents:
    entities.append(ent.text)

# set "removes" duplicates, maybe ensures no duplicates through its initialization is more correct
print(f'These are all the unique entities! {set(entities)}')

# let's count the number of adjectives (and find the rel frequency)

adjective_count = 0
for token in doc:
    if token.pos_ == "ADJ":
        adjective_count += 1

adj_rel_freq = (adjective_count/len(doc)) * 10000

print(f'The relative frequency of adjectives are {adj_rel_freq}')

# ideas include making an initial list, looping through each and calculating score
def count_occurances(document):
    noun_count = verb_count = adj_count = adv_count = 0
    for token in doc:
        match token.pos_:
            case "NOUN":
                noun_count += 1
            case "VERB":
                verb_count += 1
            case "ADJ":
                adj_count += 1
            case "ADV":
                adv_count += 1
            case _:
                continue
    return adj_count

def calculate_relative_frequency(document, count, per_how_many_words):
    return (count/len(doc)) * per_how_many_words

#dictionaries are ordered as per Python v 3.7
def count_occurances_comprehension(document):
    NOUN, VERB, ADJ, ADV = 'noun_count', 'verb_count', 'adj_count', 'adv_count'
    token_types = {
        'noun_count': 0,
        'verb_count': 0,
        'adj_count': 0,
        'adv_count': 0
        }

# can validate dict structure, probably do variable parameter (if possible :O)
    for token in doc:
        try:
            match token.pos_:
                case "NOUN":
                    token_types['noun_count'] += 1
                case "VERB":
                    token_types['verb_count'] += 1
                case "ADJ":
                    token_types['adj_count'] += 1
                case "ADV":
                    token_types['adv_count'] += 1
                case _:
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")

    
    return [calculate_relative_frequency(doc, count, 10000) for count in token_types.values()]
    
    

frequency_list = count_occurances_comprehension(doc)
print(f'This is the frequency list: {frequency_list}')


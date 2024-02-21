import pandas as pd
import spacy
import os
import sys
from charset_normalizer import from_bytes, from_path
import chardet

#probably gonna nest for loops for iterating all the files in subfolders

# load model, language class initialized through setup.sh
nlp = spacy.load("en_core_web_md")

input_folder_file_path = os.path.join("data", "input")
input_file_for_testing = os.path.join(input_folder_file_path, "b1", "0103.b1.txt")


# NEED TO CHECK FOR ENCODING 

with open (input_file_for_testing, "r", encoding="ISO-8859-1") as file:
    text = file.read()



doc = nlp(text)

# CHECK FOR ENCODING, ASSUMING UTF-8, ATTEMPT DETERMINE ENCODING
def open_text_file(filepath):
    try:
        with open (filepath, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open (filepath, "rb") as file:
            get_encoding = chardet.detect(file.read()['encoding'])
        try:
            with open (filepath, encoding=get_encoding) as file:
                return file.read()
        except:
            print("ERROR")
            break

def folder_loop(input_folder_path):
    folder_count = 0
    input_folders = os.listdir(input_folder_path)
    # loop through all folders
    for folder in input_folders:
        # get all sub folders of current folder in loop
        input_sub_folders = os.listdir(os.path.join(input_folder_path, folder))
        # loop through all sub folders
        for sub_folder in input_sub_folders:
            folder_count += 1
    print(folder_count)

def count_occurances_named_entity(document):
    person_count = 0
    org_count = 0
    loc_count = 0
    for ent in document.ents:
        try:
            match ent.label_:
                case 'PERSON':
                    person_count += 1
                case 'ORG':
                    org_count += 1
                case 'LOC':
                    loc_count += 1
        except TypeError:
            return
    print(person_count, org_count)

# ideas include making an initial list, looping through each and calculating score

def calculate_relative_frequency(document, count, per_how_many_words):
    return round((count/len(doc)) * per_how_many_words, 2)

# rewrite to take a list and create based on that
def count_occurances_type(document):
    # initialize variables to ensure consistency in references
    NOUN, VERB, ADJ, ADV, OTHER_COUNT, LOOP_COUNT = 'noun_count', 'verb_count', 'adj_count', 'adv_count', 0, 0
    token_types = {
        NOUN: 0,
        VERB: 0,
        ADJ: 0,
        ADV: 0
        }

# can validate dict structure, probably do variable parameter (if possible :O)
    for token in document:
        try:
            match token.pos_:
                case "NOUN":
                    token_types[NOUN] += 1
                    LOOP_COUNT += 1
                case "VERB":
                    token_types[VERB] += 1
                    LOOP_COUNT += 1
                case "ADJ":
                    token_types[ADJ] += 1
                    LOOP_COUNT += 1
                case "ADV":
                    token_types[ADV] += 1
                    LOOP_COUNT += 1
                case _:
                    OTHER_COUNT += 1
                    LOOP_COUNT += 1
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")
    print(f'[COMPLETE] {LOOP_COUNT} tokens have been looped through. {OTHER_COUNT} were not of the specified types.')
    return [calculate_relative_frequency(doc, count, 10000) for count in token_types.values()]

frequency_list = count_occurances_type(doc)
print(f'This is the frequency list: {frequency_list}')
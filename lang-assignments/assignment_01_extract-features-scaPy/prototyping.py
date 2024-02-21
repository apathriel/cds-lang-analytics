import pandas as pd
import spacy
import os
import sys
import chardet
import re

#probably gonna nest for loops for iterating all the files in subfolders

# load model, language class initialized through setup.sh
nlp = spacy.load("en_core_web_md")

input_folder_file_path = os.path.join("data", "input")
output_folder_file_path = os.path.join("data", "output")

def clean_text(text):
    return re.sub(r'<[^>]+>', '', text)

# CHECK FOR ENCODING, ASSUMING UTF-8, ATTEMPT DETERMINE ENCODING
def open_text_file(filepath):
    try:
        with open (filepath, "r", encoding="utf-8") as file:
            return clean_text(file.read())
    except UnicodeDecodeError:
        with open (filepath, "rb") as file:
            get_encoding = chardet.detect(file.read())
        try:
            with open (filepath, "r", encoding=get_encoding['encoding']) as file:
                return clean_text(file.read())
        except:
            print("ERROR")

def calculate_relative_frequency(document, count, per_how_many_words):
    return round((count/len(document)) * per_how_many_words, 2)

def folder_loop(input_folder_path, output_folder_path):
    folder_count = 0
    input_folders = os.listdir(input_folder_path)
    for sub_folder in input_folders:
        print("[SYSTEM] Processing subfolder...")
        input_sub_folder_path = os.path.join(input_folder_path, sub_folder)
        input_sub_folders = os.listdir(input_sub_folder_path)
        df = pd.DataFrame()
        for file in input_sub_folders:
            text = open_text_file(os.path.join(input_sub_folder_path, file))
            doc = nlp(text)
            #can merge these dicts, maybe use comprehension to overwrite, maybe do it in a function lowkey
            type_dict = count_occurances_type(doc)
            entity_dict = count_occurances_named_entity(doc)
            try:
                create_df_row = pd.DataFrame({
                    'Filename': file, 
                    'RelFreq NOUN': [type_dict['noun_count']],
                    'RelFreq VERB':[type_dict['verb_count']],
                    'RelFreq ADJ': [type_dict['adj_count']],
                    'RelFreq ADV': [type_dict['adv_count']],
                    'No. Unique PER':[entity_dict['PERSON']],
                    'No. Unique LOC': [entity_dict['LOC']],
                    'No. Unique ORG': [entity_dict['ORG']]
                    })
                df = pd.concat([df, create_df_row], ignore_index=True)
            except TypeError:
                create_df_row = pd.DataFrame({
                    'Filename': ['N/A'], 
                    'RelFreq NOUN': ['NA'],
                    'RelFreq VERB':['NA'],
                    'RelFreq ADJ': ['NA'],
                    'RelFreq ADV': ['NA'],
                    'No. Unique PER':['NA'],
                    'No. Unique LOC': ['NA'],
                    'No. Unique ORG': ['NA']
                    })
                df = pd.concat([df, create_df_row], ignore_index=True)
        df.to_csv(f"{os.path.join(output_folder_path, sub_folder + '_table' + '.csv')}", index=False)
        print('[SYSTEM] Subfolder has been processed')

def count_occurances_named_entity(document):
    # ensure unique values through sets, alternative incorporate dict structure, validate through if not in, which solution performance wise?
    PERSON, ORG, LOC, OTHER_COUNT, LOOP_COUNT = 'PERSON', 'ORG', 'LOC', 0, 0
    entity_counts = {PERSON: set(), ORG: set(), LOC: set()}

    for ent in document.ents:
        try:
            match ent.label_:
                case 'PERSON':
                    entity_counts[PERSON].add(ent.text)
                case 'ORG':
                    entity_counts[ORG].add(ent.text)
                case 'LOC':
                    entity_counts[LOC].add(ent.text)
                case _:
                    OTHER_COUNT += 1
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")
        
        #dictionary comprehension - create new dict applying len function to each key
        """ print(f'[COMPLETE] {LOOP_COUNT} entities have been looped through. {OTHER_COUNT} were not associated with the specified labels') """
        new_dict = {label: len(count) for label, count in entity_counts.items()}
        print(new_dict)
        return new_dict
        

def count_occurances_type(document):
    NOUN, VERB, ADJ, ADV, OTHER_COUNT, LOOP_COUNT = 'noun_count', 'verb_count', 'adj_count', 'adv_count', 0, 0
    pos_counts = {
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
                    pos_counts[NOUN] += 1
                    LOOP_COUNT += 1
                case "VERB":
                    pos_counts[VERB] += 1
                    LOOP_COUNT += 1
                case "ADJ":
                    pos_counts[ADJ] += 1
                    LOOP_COUNT += 1
                case "ADV":
                    pos_counts[ADV] += 1
                    LOOP_COUNT += 1
                case _:
                    OTHER_COUNT += 1
                    LOOP_COUNT += 1
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")
    
    """ print(f'[COMPLETE] {LOOP_COUNT} tokens have been looped through. {OTHER_COUNT} were not of the specified types.') """
    return {key: calculate_relative_frequency(document, value, 10000) for key, value in pos_counts.items()}

# --------------------------------------------------
""" frequency_list = count_occurances_type(doc)
print(f'This is the frequency list: {frequency_list}')

entity_occurances = count_occurances_named_entity(doc)
print(f'This is the entity occurances {entity_occurances}') """

""" if __name__ == "__main__":
    main() """

folder_loop(input_folder_file_path, output_folder_file_path)
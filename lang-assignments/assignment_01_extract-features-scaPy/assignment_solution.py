import pandas as pd
import spacy
import os
import chardet
import re

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
    # explitize in readme
    return round((count/len(document)) * per_how_many_words, 2)

def process_folders(input_path, output_path, model):
    input_folders = os.listdir(input_path)

    for sub_folder in input_folders:
        print(f"[SYSTEM] Processing subfolder {sub_folder}")
        input_sub_folder_path = os.path.join(input_path, sub_folder)
        input_sub_folders = os.listdir(input_sub_folder_path)
        df_list = []  # create list for storing dataframes, append each, then concatenate all at once

        for file in input_sub_folders:
            print(f"[SYSTEM] Processing file: {file}")
            text = open_text_file(os.path.join(input_sub_folder_path, file))
            doc = model(text)
            values_dict = count_occurances_type(doc) | count_occurances_named_entity(doc)
            create_df_row = pd.DataFrame({
                'Filename': [file], 
                'RelFreq NOUN': [values_dict['noun_count']],
                'RelFreq VERB':[values_dict['verb_count']],
                'RelFreq ADJ': [values_dict['adj_count']],
                'RelFreq ADV': [values_dict['adv_count']],
                'No. Unique PER':[values_dict['PERSON']],
                'No. Unique LOC': [values_dict['LOC']],
                'No. Unique ORG': [values_dict['ORG']]
                })
            df_list.append(create_df_row)  # append the DataFrame a list

        df = pd.concat(df_list, ignore_index=True)  # concatenate all the DataFrames at once
        df.to_csv(f"{os.path.join(output_path, sub_folder + '_table' + '.csv')}", index=False)
        print('[SYSTEM] Subfolder has been processed')

def count_occurances_named_entity(document):
    # ensure unique values through sets, alternative incorporate dict structure, validate through if not in, which solution performance wise?
    PERSON, ORG, LOC = 'PERSON', 'ORG', 'LOC'
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
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")
        
    #dictionary comprehension - create new dict applying len function to each key
    return {label: len(count) or 0 for label, count in entity_counts.items()}


def count_occurances_type(document):
    NOUN, VERB, ADJ, ADV = 'noun_count', 'verb_count', 'adj_count', 'adv_count'
    pos_counts = {
        NOUN: 0,
        VERB: 0,
        ADJ: 0,
        ADV: 0
    }

    for token in document:
        try:
            match token.pos_:
                case "NOUN":
                    pos_counts[NOUN] += 1
                case "VERB":
                    pos_counts[VERB] += 1
                case "ADJ":
                    pos_counts[ADJ] += 1
                case "ADV":
                    pos_counts[ADV] += 1
                case _:  # default case
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")

    return {key: calculate_relative_frequency(document, value, 10000) for key, value in pos_counts.items()}

if __name__ == "__main__":
    # load model, language class initialized through setup.sh
    # select file, inquirer-inspired 
    nlp = spacy.load("en_core_web_md")
    input_folder_path = os.path.join("data", "input")
    output_folder_path = os.path.join("data", "output")
    
    process_folders(input_folder_path, output_folder_path, nlp)
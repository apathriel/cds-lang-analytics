from pathlib import Path
from typing import Dict, Union

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm

from cli_utilities import parse_cli_arguments
from data_processing_utilities import export_df_as_csv, load_text_file
from utilities import get_logger

logger = get_logger(__name__)


def calculate_relative_frequency(
    document: Doc,
    count: int,
    per_how_many_words: int = 10000,
    remove_punctuation: bool = True,
) -> float:
    """
    Calculates the relative frequency of a certain type of token (pos) in document loaded through a spaCy model, per a certain number of words.

    Parameters:
        document (List) (tokens processed by spaCy model): The document (doc) class to analyze, represented as a list of tokens.
        count (int): The number of occurrences of the token type in the document.
        per_how_many_words (int, optional): The number of words per which the relative frequency is calculated.
        remove_punctuation (bool, optional): Whether to remove punctuation tokens from the document before calculating the relative frequency.

    Returns:
        float: The relative frequency of the token in the document, rounded to two decimal places.
    """
    total_tokens = len(document)
    if remove_punctuation:
        non_punct_tokens = sum(1 for token in document if not token.is_punct)
        relative_frequency = non_punct_tokens / total_tokens * per_how_many_words
    else:
        relative_frequency = count / total_tokens * per_how_many_words

    return round(relative_frequency, 2)


def extract_linguistic_information_pipeline(
    input_path: Path,
    output_path: Path,
    model: Union[Language, str],
    remove_punctuation: bool = False,
) -> str:
    """
    Processes all text files in the given input folder and its subfolders, and writes the results to CSV files in the designated output folder.

    This function iterates over all subfolders in the input folder. For each text file in each subfolder, it opens the file,
    processes the text using the given spaCy model, and calculates various statistics about the text.

    Parameters:
        input_path (str): The path to the input folder.
        output_path (str): The path to the output folder.
        model (spacy.lang): The spaCy language model to use for NLP.
        remove_punctuation (bool, optional): Whether to remove punctuation tokens from the documents before calculating the relative frequency.

    Returns:
        str: Message indicating successful completion of the operation.

    """
    if not any(input_path.iterdir()):
        return logger.error("No subfolders found in the input folder.")

    for sub_directory in tqdm(input_path.iterdir(), desc="Processing subfolders"):
        if sub_directory.is_dir():
            dataframes_to_concatenate = []  # create list for storing dataframes, append each, then concatenate all at once
            for file in tqdm(
                sub_directory.iterdir(),
                desc=f"Processing files in directory {sub_directory.name}",
            ):
                text = load_text_file(file)
                doc = model(text)

                values_dict = calculate_token_type_occurrences(
                    doc, remove_punctuation
                ) | calculate_named_entity_occurrences(doc)
                create_df_row = pd.DataFrame(
                    {
                        "Filename": [file.name],
                        "RelFreq NOUN": [values_dict["noun_count"]],
                        "RelFreq VERB": [values_dict["verb_count"]],
                        "RelFreq ADJ": [values_dict["adj_count"]],
                        "RelFreq ADV": [values_dict["adv_count"]],
                        "No. Unique PER": [values_dict["PERSON"]],
                        "No. Unique LOC": [values_dict["LOC"]],
                        "No. Unique ORG": [values_dict["ORG"]],
                    }
                )
                dataframes_to_concatenate.append(
                    create_df_row
                )  # append the DataFrame to a list

            df = pd.concat(
                dataframes_to_concatenate, ignore_index=True
            )  # concatenate all the DataFrames at once
            df = df.sort_values("Filename")  # sort the DataFrame by filename
            export_df_as_csv(df, output_path, f"{sub_directory.name}_table.csv")  # export the DataFrame to a CSV file
        else:
            logger.error(f"File {sub_directory.name} is not a directory.")
            continue

    return "Dataframes concatenated and exported successfully!"
    


def calculate_named_entity_occurrences(document: Doc) -> Dict[str, int]:
    """
    Counts the number of unique occurrences of certain types of named entities in a document.

    Parameters:
        document (spacy.doc): The document to analyze, represented as a spaCy Doc object.

    Returns:
        dict: A dictionary where the keys are named entity labels ('PERSON', 'ORG', 'LOC') and the values are the counts of unique named entities of each type.

    Raises:
        TypeError: If a named entity's label is not a string.
        KeyError: If a named entity's label is not one of the specified types.
    """
    PERSON, ORG, LOC = "PERSON", "ORG", "LOC"
    entity_counts = {PERSON: set(), ORG: set(), LOC: set()}

    for ent in document.ents:
        try:
            match ent.label_:
                case "PERSON":
                    entity_counts[PERSON].add(ent.text)
                case "ORG":
                    entity_counts[ORG].add(ent.text)
                case "LOC":
                    entity_counts[LOC].add(ent.text)
                case _:
                    continue
        except TypeError:
            logger.error("TypeError - Value is of wrong type")
        except KeyError:
            logger.error("KeyError -  Key does not exist :(")

    # dictionary comprehension - create new dict applying len function to each key
    return {label: len(count) or 0 for label, count in entity_counts.items()}


def calculate_token_type_occurrences(
    document: Doc, remove_punctuation: bool = False
) -> Dict[str, float]:
    """
    Counts the occurrences of certain types of tokens in a spacy document object.

    Parameters:
        document (spacy.doc): The document to analyze, represented as a spaCy Doc object.

    Returns:
        dict: A dictionary where the keys are token types ('noun', 'verb', 'adj', 'adv') and the values are the counts of tokens of each type.
    """
    NOUN, VERB, ADJ, ADV = "noun_count", "verb_count", "adj_count", "adv_count"
    pos_counts = {NOUN: 0, VERB: 0, ADJ: 0, ADV: 0}

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
                case _:
                    continue
        except TypeError:
            logger.error("[TypeError] Value is of wrong type")
        except KeyError:
            logger.error("[KeyError] Key does not exist :(")

    return {
        key: calculate_relative_frequency(document, value, 10000, remove_punctuation)
        for key, value in pos_counts.items()
    }


def main():
    # Get the command-line arguments
    cli_args = parse_cli_arguments()

    # Try to load the spaCy model, download it if it's not found
    try:
        nlp = spacy.load(cli_args.model)
    except OSError:
        logger.error(f"{cli_args.model} not found. Attempting to download model...")
        spacy.cli.download(cli_args.model)
        nlp = spacy.load(cli_args.model)

    # Initialize the input and output folder paths. File paths are expected to be relative to the src folder. Within project root.
    input_folder_path = Path(__file__).parent / ".." / cli_args.input_path
    output_folder_path = Path(__file__).parent / ".." / cli_args.output_path

    # Extract linguistic information from the text files
    extract_linguistic_information_pipeline(
        input_path=input_folder_path, 
        output_path=output_folder_path,
        model=nlp
    )


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path
from typing import Any


import pandas as pd

from utils.cli_utils import get_cli_args
from utils.data_processing_utils import *
from utils.model_utils import load_gensim_model, get_word_embeddings
from utils.logging_utils import get_logger
from utils.utilities import (
    calculate_percentage_2_integers,
    extract_nth_element_from_list_of_tuples,
)

logger = get_logger(__name__)



def check_if_artist_in_dataset(
    df: pd.DataFrame, artist: str, column_filter_val: str = "artist"
) -> bool:
    """
    Check if the given artist is present in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset to search in.
        artist (str): The artist to check for.
        column_filter_val (str, optional): The column name to filter on. Defaults to "artist".

    Returns:
        bool: True if the artist is present in the dataset, False otherwise.
    """
    return artist in get_unique_row_values_by_column(df, column_filter_val)


def print_query_results(
    percent: float,
    artist: str,
    query: str,
    output_path: Path,
    output_file: str,
    save_to_csv: bool = True,
) -> None:
    """
    Prints the query results and optionally saves them to a CSV file.

    Parameters:
        percent (float): The percentage of songs containing the words related to the query.
        artist (str): The artist associated with the query results.
        query (str): The query for which the results are being printed.
        output_path (Path): The path to the directory where the output file will be saved.
        output_file (str): The name of the output file.
        save_to_csv (bool, optional): Whether to save the results to a CSV file. Defaults to True.
    """
    logger.info(f"{percent}% of {artist} songs contain the words related to {query}")

    if save_to_csv:
        logger.info(f"Saving results to {output_file}...")

        df = load_existing_df_or_create_new_df(
            output_path / output_file, ["percent", "artist", "query"]
        )

        # Check if the artist with the specific query already exists
        existing_rows = df.loc[(df['artist'] == artist) & (df['query'] == query)]

        if existing_rows.empty:
            new_row = pd.DataFrame(
                {"artist": [artist], "percent": [percent], "query": [query]}
            )
            df = pd.concat([df, new_row], ignore_index=True)

            df.reset_index(drop=True, inplace=True)

            write_csv_to_file(df, output_path, output_file, remove_empty_rows=True)
            logger.info(f"Results saved to {output_file}")
        else:
            logger.info(f"Artist {artist} with query {query} already exists in the DataFrame.")


def validate_artist_input(
    df: pd.DataFrame, parser: ArgumentParser, artist_to_check: str
) -> pd.DataFrame:
    """
    Validates the artist input by checking if it exists in the dataset.

    Parameters:
        df (pd.DataFrame): The dataset to validate against.
        parser (ArgumentParser): The argument parser object.
        artist_to_check (str): The artist name to validate.

    Returns:
        pd.DataFrame: The filtered dataframe containing rows with the specified artist.

    Raises:
        SystemExit: If the artist is not found in the dataset.

    """
    logger.info(
        f"Validating artist input: {artist_to_check}, checking presence in dataset..."
    )
    if not check_if_artist_in_dataset(df, artist_to_check):
        closest_match = find_closest_match(df, "artist", artist_to_check)
        closest_match_string = (
            f"Did you perhaps mean: {''.join(find_closest_match(df, 'artist', artist_to_check))}?"
            if closest_match
            else "No similar artist found in dataset."
        )
        parser.exit(
            1,
            message=f"[PARSER_ERROR] Artist '{artist_to_check}' not found in dataset. {closest_match_string}",
        )
    logger.info(f"Artist '{artist_to_check}' found in dataset!")
    return filter_df_rows_by_column_value(df, "artist", artist_to_check)


def main():
    parser, cli_args = get_cli_args()

    gensim_model_name = cli_args.model if cli_args.model else "glove-wiki-gigaword-50"
    input_dataset_path = (
        Path(__file__).parent / ".." / "in" / "spotify_million_dataset.csv"
    )
    output_data_path = Path(__file__).parent / ".." / "out"

    model = load_gensim_model(gensim_model_name)

    word_embeddings = extract_nth_element_from_list_of_tuples(
        get_word_embeddings(model, parser, cli_args.query), n=0
    )

    df = load_csv_to_df(input_dataset_path)

    for artist in cli_args.artist:
        df_by_artist = validate_artist_input(df, parser, artist)

        songs_containing_concept = filter_df_by_term_occurance(
            df_by_artist, "text", word_embeddings
        )

        print_query_results(
            calculate_percentage_2_integers(
                get_num_rows(songs_containing_concept), get_num_rows(df_by_artist)
            ),
            artist,
            cli_args.query,
            output_data_path,
            "query_results.csv",
            cli_args.save,
        )


if __name__ == "__main__":
    main()

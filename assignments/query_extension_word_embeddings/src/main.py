import os

from data_processing_utils import *
from model_utils import load_gensim_model, get_word_embeddings
import pandas as pd
from query_artist_corpus import get_cli_args
from utilities import calculate_percentage_2_integers, extract_nth_element_from_list_of_tuples

def check_if_artist_in_dataset(df, artist, column_filter_val='artist') -> bool:
    return artist in get_unique_row_values_by_column(df, column_filter_val)

def load_existing_df_or_create_new_df(data_path, columns):
    if os.path.isfile(data_path):
        return load_csv_to_df(data_path)
    else:
        return pd.DataFrame(columns=columns)

def print_query_results(percent, artist, query, output_path, output_file, save_to_csv=True):
    print(f'[SYSTEM] {percent}% of {artist} songs contain the words related to {query}')
    
    if save_to_csv:
        print(f"[SYSTEM] Saving results to {output_file}...")

        df = load_existing_df_or_create_new_df(os.path.join(output_path, output_file), ['percent', 'artist', 'query'])
        new_row = pd.DataFrame({'artist': [artist], 'percent': [percent], 'query': [query]})
        df = pd.concat([df, new_row], ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        
        write_csv_to_file(df, os.path.join(output_path, output_file))
        print(f"[SYSTEM] Results saved to {output_file}")

def validate_artist_input(df, parser, artist_to_check):
    print(f"[SYSTEM] Validating artist input: {artist_to_check}, checking presence in dataset...")
    if not check_if_artist_in_dataset(df, artist_to_check):
      closest_match = find_closest_match(df, "artist", artist_to_check)
      closest_match_string = f"Did you perhaps mean: '{''.join(find_closest_match(df, "artist", artist_to_check))}'?" if closest_match else 'No similar artist found in dataset.'
      parser.exit(1, message=f"[PARSER_ERROR] Artist '{artist_to_check}' not found in dataset. {closest_match_string}")
    return filter_df_rows_by_column_value(df, "artist", artist_to_check)

def main():
    parser, cli_args = get_cli_args()

    gensim_model_name = cli_args.model if cli_args.model else 'glove-wiki-gigaword-50'
    input_dataset_path = os.path.join(os.path.dirname(__file__), "..", "in", 'spotify_million_dataset.csv')
    output_data_path = os.path.join(os.path.dirname(__file__), "..", "out")
    
    model = load_gensim_model(gensim_model_name)
    word_embeddings = extract_nth_element_from_list_of_tuples(get_word_embeddings(model, parser, cli_args.query), n=0)

    df = load_csv_to_df(input_dataset_path)
    df_by_artist = validate_artist_input(df, parser, cli_args.artist)

    songs_containing_concept = filter_df_by_term_occurance(df_by_artist, "text", word_embeddings)
    
    print_query_results(
                        calculate_percentage_2_integers(get_num_rows(songs_containing_concept), get_num_rows(df_by_artist)), 
                        cli_args.artist, 
                        cli_args.query,
                        output_data_path,
                        'query_results.csv',
                        cli_args.save
                        )

if __name__ == "__main__":
    main()
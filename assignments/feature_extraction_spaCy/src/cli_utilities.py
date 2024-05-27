import argparse

def parse_cli_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process text files and extract linguistic information.")

    # Add the arguments
    parser.add_argument('-m', '--model', type=str, help='The spaCy language model to use for NLP.', default="en_core_web_md")
    parser.add_argument('-i', '--input_path', type=str, help='The path to the input folder.', default="in")
    parser.add_argument('-o', '--output_path', type=str, help='The path to the output folder.', default="out")

    # Parse the arguments
    args = parser.parse_args()

    return args
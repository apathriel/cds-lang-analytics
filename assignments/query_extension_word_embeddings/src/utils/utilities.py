import string

def calculate_percentage_2_integers(num1: int, num2: int, decimal_places=2):
    if not isinstance(num1, int) or not isinstance(num2, int):
        raise TypeError("Both num1 and num2 must be integers")
    if not isinstance(decimal_places, int):
        raise TypeError("decimal_places must be an integer")
    if num2 == 0:
        raise ValueError("num2 must not be zero")
    return round((num1 / num2) * 100, decimal_places)

def extract_nth_element_from_list_of_tuples(list_of_tuples: list, n: int) -> list:
    return [element[n] for element in list_of_tuples]

def convert_string_to_lower_case(string: str) -> str:
    return string.lower()

def remove_punctuation_from_list(list_of_strings: list) -> list:
    return [element for element in list_of_strings if not any(char in string.punctuation for char in element)]

def escape_punctuation_in_list(list_of_strings: list) -> list:
    return [''.join(f'\\{char}' if char in string.punctuation else char for char in element) for element in list_of_strings]
import string
from typing import Any, List, Tuple


def calculate_percentage_2_integers(
    num1: int, num2: int, decimal_places: int = 2
) -> float:
    """
    Calculates the percentage of num1 relative to num2.

    Args:
        num1 (int): The numerator.
        num2 (int): The denominator.
        decimal_places (int, optional): The number of decimal places to round the result to. Defaults to 2.

    Returns:
        float: The calculated percentage.

    Raises:
        TypeError: If num1 or num2 is not an integer.
        TypeError: If decimal_places is not an integer.
        ValueError: If num2 is zero.
    """
    if not isinstance(num1, int) or not isinstance(num2, int):
        raise TypeError("Both num1 and num2 must be integers")
    if not isinstance(decimal_places, int):
        raise TypeError("decimal_places must be an integer")
    if num2 == 0:
        raise ValueError("num2 must not be zero")
    if decimal_places < 0:
        raise ValueError("decimal_places must be greater than or equal to zero")
    return round((num1 / num2) * 100, decimal_places)


def extract_nth_element_from_list_of_tuples(
    list_of_tuples: List[Tuple[Any, ...]], n: int
) -> List[Any]:
    """
    Extracts the nth element from each tuple in a list of tuples.

    Parameters:
        list_of_tuples (List[Tuple[Any, ...]]): The list of tuples from which to extract the nth element.
        n (int): The index of the element to extract from each tuple.

    Returns:
        List[Any]: A list containing the nth element from each tuple.
    """
    return [element[n] for element in list_of_tuples]


def convert_string_to_lower_case(string: str) -> str:
    """
    Converts a string to lowercase.

    Parameters:
        string (str): The input string.

    Returns:
        str: The string converted to lowercase.
    """
    return string.lower()


def remove_punctuation_from_list(list_of_strings: List[str]) -> List[str]:
    """
    Removes punctuation from a list of strings.

    Parameters:
        list_of_strings (List[str]): A list of strings.

    Returns:
        List[str]: A new list of strings with punctuation removed.
    """
    punctuation_set = set(string.punctuation)
    return [
        element
        for element in list_of_strings
        if not any(char in punctuation_set for char in element)
    ]


def escape_punctuation_in_list(list_of_strings: List[str]) -> List[str]:
    """
    Escapes punctuation characters in a list of strings.

    Parameters:
        list_of_strings (List[str]): A list of strings.

    Returns:
        List[str]: A new list of strings with punctuation characters escaped.

    """
    return [
        "".join(f"\\{char}" if char in string.punctuation else char for char in element)
        for element in list_of_strings
    ]

def recursively_find_in_dict(data, target_field):
    """
    Recursively searches for a field in a nested dictionary.

    Args:
        data: The dictionary to search.
        target_field: The field to find.

    Returns:
        The value of the field if found, otherwise None.
    """
    if isinstance(data, dict):
        if target_field in data:
            return data[target_field]
        else:
            for key, value in data.items():
                result = recursively_find_in_dict(value, target_field)
                if result is not None:
                    return result
    return None


def flatten_dict(data):
    """
    Recursively traverses a nested dictionary and outputs its keys and values in a sorted list.

    Args:
        data: The input dictionary

    Returns:
        A sorted tuple of keys and values
    """
    output_list = []
    if isinstance(data, dict):
        for key, value in data.items():
            output_list.append(tuple([key, flatten_dict(value)]))
        output_list = sorted(output_list)
        return tuple(output_list)
    elif isinstance(data, list):
        output_list = [flatten_dict(x) for x in data]
        return tuple(output_list)
    else:
        return data

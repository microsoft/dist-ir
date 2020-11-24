def map_values(values, value_name_map, value_map, device):
    """Tries to resolve the given values according to the specified maps."""
    if value_name_map is None:
        return values
    mapped_values = []
    for value in values:
        if value.name in value_name_map[device]:
            mapped_value_name = value_name_map[device][value.name]
            mapped_values.append(value_map[mapped_value_name])
        else:
            mapped_values.append(value)
    return mapped_values

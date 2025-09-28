def map_label(original_class):
    biodegradable = ['paper', 'cardboard']
    recyclable = ['metal', 'glass']
    non_recyclable = ['trash', 'plastic']

    if original_class in biodegradable:
        return 0  # Biodegradable
    elif original_class in recyclable:
        return 1  # Recyclable
    else:
        return 2  # Non-Recyclable

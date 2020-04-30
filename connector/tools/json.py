import json


def save_json_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

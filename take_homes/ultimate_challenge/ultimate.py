import json

import pandas as pd


def read_file(filename):
    # Open and read the file
    with open(filename, 'r') as file:
        data = file.read()

    # Parse the JSON-like data into a Python dictionary
    try:
        data_dict = json.loads(data)
        # login_times = data_dict.get('login_time', [])
        df = pd.DataFrame(data_dict)
        return df
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        login_times = []
    return login_times


if __name__ == '__main__':
    p = pd.read_json('logins.json')
    print("")
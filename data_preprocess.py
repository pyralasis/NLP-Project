#this file is intended to 
import json

def merge_datasets(data_paths, output_path):
    all_data = []
    
    # Load and merge datasets
    for path in data_paths:
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                all_data.extend(data)
                print(f"Loaded {len(data)} entries from {path}")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {path}")

    # Write merged data to the specified output file
    with open(output_path, 'w') as output_file:
        json.dump(all_data, output_file)#, indent=4)
        print(f"Merged dataset saved to {output_path}")

# dataset paths # CHANGE THIS TO INCLUDE DATA YOU WANT
data_paths = [
    "./DATA/og_data/mpqa/train.json",#mpqa2.0 if time look in to 3.0 
    "./DATA/og_data/opener_en.json"#Opener_en
    #Darmstadt_unis # TO-DO
]

# output file path and name
output_path = "DATA/test_data/WIP_merged_english_data_WIP.json"

# Call the merge function
merge_datasets(data_paths, output_path)


# syntehtic gerneration TO-DO? bert may take care of this: this will need to take in to account spans changing and targets? complex problem, could use other LLM. 
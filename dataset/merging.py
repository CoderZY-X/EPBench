"""
Merge all CSV files in subdirectories of the specified directory
"""
import pandas as pd
import os

def merge_csv_files(input_dir: str, output_file: str):


    dataframes = []


    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)


    merged_df = pd.concat(dataframes, ignore_index=True)


    merged_df.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"Merging complete. Saved as: {output_file}")

def main():
    input_dir = r".../train"
    output_file = r".../train.csv"

    merge_csv_files(input_dir, output_file)

if __name__ == "__main__":
    main()
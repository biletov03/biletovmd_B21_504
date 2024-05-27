import csv

def hex_to_char(hex_code):
    try:
        char = chr(int(hex_code, 16))
        return char
    except ValueError:
        return hex_code

def convert_symbols(input_path, output_path):
    with open(input_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)

        for row in rows:
            new_row = []
            for i, value in enumerate(row):
                if i % 2 == 1:
                    new_row.append(hex_to_char(value))
                else:
                    new_row.append(value)
            writer.writerow(new_row)


if __name__ == "__main__":
    input_csv = 'output/new_4_letter_features.csv'
    output_csv = 'output/new_4_letter_features.csv'
    convert_symbols(input_csv, output_csv)

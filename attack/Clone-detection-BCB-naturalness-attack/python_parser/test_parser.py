import argparse
import csv
import sys
sys.path.append('.')
sys.path.append('../')
from run_parser import get_identifiers
path = 'parser_folder/my-languages.so'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    args = parser.parse_args()
    f_write = open('attack_c_bible.csv', 'w')
    writer = csv.writer(f_write)
    with open('attack_result_c.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                write_data = ['Original Code', 'Extracted Variable']
                writer.writerow(write_data)
                line_count += 1
            elif line_count == 50:
                break
            else:
                data, _ = get_identifiers(row[0], args.lang)
                write_data = [row[0], data]
                writer.writerow(write_data)
                line_count += 1
        print(f'Processed {line_count} lines.')

    f_write.close()


if __name__ == '__main__':
    main()


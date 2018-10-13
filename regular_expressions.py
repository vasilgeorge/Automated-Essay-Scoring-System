import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("essays_dataset_input")
parser.add_argument("essays_dataset_output")
args = parser.parse_args()

inp_file = args.essays_dataset_input
with open(inp_file) as input:
    text = input.read()

new = text.replace("\t", ' ')   # Remove tabs
new_t = new.replace("\r", ' ')  # Remove line breaks
new_text = new_t.replace("\n", ' ') # Remove line breaks

essay = re.sub(r'I[0-9]{6}\s+[0-9]{1}', '@!@', new_text)    # Replace any sequence of characters such as I0xxx with the separator @!@
essay1 = re.sub(r'\s+', ' ', essay)                         # Replace multiple spaces with one space
essay2  = re.sub(r'@!@[0-9].0', '@!@', essay1)              # Replace score, since all the essays in a file have the same score
essay3 = re.sub(r'([0-9]{3}:[0-9]{5}-ES[^@])', r'\1 @!@', essay2)   # Insert separator after the essay id
essay4 = re.sub(r'@!@\Z', '', essay3)                       # Remove the separator if it is in the end of a file/string
essay5 = re.sub(r'(@!@\s*)([0-9]{3}:)', r'\1\n\2', essay4)  # Insert a line break if the essay ends and a new id starts
essay6 = re.sub(r'@!@\s*\n', '\n', essay5)                  # Fix some minor errors with line breaks

out_file = args.essays_dataset_output
with open(out_file, 'w') as out:
    out.write(essay6)

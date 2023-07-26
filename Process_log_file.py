def match_and_collect_rows(file1, file2, output_file):
    # Read the matching strings into a list
    with open(file1, 'r') as f:
        match_strings = [line.strip() for line in f]

    # Process the second file
    with open(file2, 'r') as f:
        lines = f.readlines()

    # Prepare the output
    output_lines = []
    for i in range(len(lines)):
        for match in match_strings:
            if match in lines[i]:
                # Append the matched line and the next 4 lines
                output_lines.extend(lines[i:i+5])
                break

    # Write the output to the file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

# Use the function
match_and_collect_rows('review_names.txt', 'epoch_10_log.txt', 'output_log.txt')

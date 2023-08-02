# Step 1: Read the order file
with open('/data/wesley/Code/review_names.txt', 'r') as f:
    order = [line.strip() for line in f]

# Step 2: Read the information file
with open('/data/wesley/Code/ty_train_output_log.txt', 'r') as f:
    lines = f.readlines()

info_dict = {}
for i in range(0, len(lines), 6):
    key = lines[i].split()[-1]  # Get the Crystal ID, assumes it's the last word in the line
    key = key.split('_')[0]  # Removes the timestamp and label information, assumes Crystal ID doesn't have underscore
    value = lines[i+1:i+6]
    info_dict[key] = value

# Step 3: Write to the new file
with open('TY_ordered_info.txt', 'w') as f:
    for key in order:
        if key in info_dict:
            f.write('Crystal ID: ' + key + '\n')
            for line in info_dict[key]:
                f.write(line)

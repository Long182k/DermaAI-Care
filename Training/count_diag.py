import csv
from collections import Counter

# Path to your CSV file
csv_file = "data/ISIC_2020_Training_GroundTruth_v2.csv"

# Initialize a Counter to keep track of diagnosis types
diagnosis_counter = Counter()

# Open the CSV file and read it
with open(csv_file, mode='r') as file:
    csv_reader = csv.DictReader(file)
    
    # Iterate through each row in the CSV file
    for row in csv_reader:
        # Get the diagnosis type from the row
        diagnosis = row['diagnosis']
        
        # Increment the counter for the diagnosis
        diagnosis_counter[diagnosis] += 1

# Print the count of each diagnosis type
for diagnosis, count in diagnosis_counter.items():
    print(f"{diagnosis}: {count}")

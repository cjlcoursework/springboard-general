import csv
import datetime
import numpy as np

baseline_tolls_per_hour = 10
starting_date = datetime.datetime(2023, 3, 1, 0, 0, 0)
reimbursement_program_date = datetime.datetime(2023, 4, 1, 0, 0, 0)
ending_date = datetime.datetime(2023, 6, 1, 0, 0, 0)
growth_rate = 0.005  # Adjust the growth rate as needed
level_date = datetime.datetime(2023, 5, 1, 0, 0, 0)

def randomized_value(static_integer):
    min_value = static_integer - 1
    max_value = static_integer + 1
    random_variation = np.random.randint(min_value, max_value + 1)
    return static_integer + random_variation

def generate_traffic():
    current_hour = starting_date

    while current_hour < ending_date:
        if current_hour < reimbursement_program_date:
            traffic = randomized_value(baseline_tolls_per_hour)
        else:
            # Calculate the time delta in hours from reimbursement_program_date
            time_delta_hours = (current_hour - reimbursement_program_date).total_seconds() / 3600

            # Apply exponential growth to the baseline tolls
            traffic = int(baseline_tolls_per_hour * (1 + growth_rate) ** time_delta_hours)

        yield current_hour, traffic
        current_hour += datetime.timedelta(hours=1)

if __name__ == '__main__':
    with open('traffic_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Traffic'])

        for timestamp, traffic in generate_traffic():
            csv_writer.writerow([timestamp, traffic])

    print("Data with exponential growth has been written to 'traffic_data.csv'.")




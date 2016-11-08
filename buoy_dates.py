import matplotlib.pyplot as plt
from pylab import savefig


def extract_dates(filename):
    start_dates = []
    end_dates = []

    with open(filename, 'r') as handle:
        for line in handle:
            # extract each piece of data from file
            ID, file_name, first_line, last_line, sd_year, sd_month, sd_day, sd_hour, ed_year, ed_month, ed_day, ed_hour = (x for x in line.split())
            # append each buoy start year to list of start dates
            start_dates.append(int(sd_year))
            # append each buoy end year to list of end dates
            end_dates.append(int(ed_year))
    return start_dates, end_dates     


def get_date_ranges(start_dates, end_dates):
    date_ranges = []*len(start_dates)

    # for each buoy in start dates, append the number of years the buoy was alive
    # (added 1 to account for buoys who were only alive for 1 year)
    for i in range(len(start_dates)):
        date_ranges.append(end_dates[i] - start_dates[i] + 1)
    return date_ranges


def count_buoys(start_dates, date_ranges):
    # bins (i.e. each year for which buoy data exists)
    x_values = sorted(set(start_dates))

    # create dictionary of years in x_values with keys set to 0
    year_counts = {}
    
    # set initial counts for each year to zero
    for item in x_values:
        year_counts[item] = 0
   
    # for each buoy in start dates, and for each year this buoy was alive,
    # add to a counter for each year
    n = 0
    for element in start_dates:
        for k in range(date_ranges[n]):
            # add one to each year's key 
            if element + k in year_counts:
                year_counts[element + k] += 1
        n += 1
    return x_values, year_counts


if __name__ == '__main__':
    filename = 'buoy_lookup_table_new.dat'
    start_dates, end_dates = extract_dates(filename)
    date_ranges = get_date_ranges(start_dates, end_dates)
    x_values, year_counts = count_buoys(start_dates, date_ranges)

    # Print and plot buoy data
    print('YEAR | # BUOYS')
    print('-----|--------')
    for year in x_values:
        print(year, '|', year_counts[year])
        plt.plot(year, year_counts[year], 'b*')
    plt.title('Number of Buoys Collecting Data in Each Year')
    plt.xlabel('Years')
    plt.ylabel('# Buoys')
    savefig('buoys_per_year.png', bbox_inches='tight')

if __name__ == '__main__':
    # =========================================================================
    # Experiment 1: Seasons
    # =========================================================================    
    
    # data_dir = '/home/hans-werner/Dropbox/work/projects/drifters/data/buoydata/'
    data_dir = os.getcwd() + '/data/'
    file_1 = 'buoydata_1_5000.dat'
    file_2 = 'buoydata_5001_10000.dat'
    file_3 = 'buoydata_10001_dec15.dat'
    file_list = [file_1, file_2, file_3]
    file_list = [file_1]
    spatial_range = {'lon_min': -180., 'lon_max': 180., 
                     'lat_min': -78., 'lat_max': 90.}
    cellwidth = 1.
    start_date = datetime(1979,2,15,0)
    end_date = datetime(1989,12,31,23,59)
    date_range = [start_date, end_date]
    dt = pd.Timedelta('30 days')
    
    # Inputs
    lookup_file = 'lookup_table_datetime.dat'
    print('Loading lookup table')
    lookup = load_lookup_table(data_dir, lookup_file)
    start_date = lookup.birth_date.min().to_datetime()
    end_date = lookup.death_date.max().to_datetime()
    n_season_list = [12,6,4]
    
    # Form matrices
    print('Forming matrices')
    A = [[],[],[]]
    season_counter = 0 
    for n_seasons in n_season_list:
        file_counter = 0
        for file_name in file_list:
            print('n_seasons:', n_seasons)
            print('file:', file_name)
            print('Building transit matrix...')
            Atmp = build_transit_matrix(data_dir, file_name, spatial_range, 
                                        cellwidth, date_range, n_seasons, dt)
            if file_counter == 0:
                A[season_counter] = Atmp
            else:
                for i in range(n_seasons):
                    A[season_counter][i] = \
                        combine_transit_matrices(A[season_counter][i],Atmp[i])
        season_counter += 1
    
    # Combine them into yearly matrices
    print('Combining into yearly matrices')
    Ay = [[],[],[]]
    sc = 0
    for n_seasons in n_season_list:
        for i in range(n_seasons):
             if i == 0:
                 Ay[sc] = A[sc][i].tocsr()
             else:
                 Ay[sc] = Ay[sc].dot(Ay[sc].tocsr())
    sc += 1

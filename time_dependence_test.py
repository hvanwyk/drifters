if __name__ == '__main__':
    # =========================================================================
    # Experiment 2: Time depence
    # =========================================================================
    
    #combine_transit_matrices(A[0], A[1],saveas='new/A12.mtx')
    
    # Normalize by row sums
    A0 = A[0].tocsr()
    row_sums = np.array(A0.sum(axis=1))[:,0]
    row_indices, col_indices = A0.nonzero()
    A0 /= row_sums[row_indices]
    
    # Extract states with nonzero rows
    
    # Compare dominant eigenvectors of 6 seasons
    
    for s in range(6):
        # Compute dominant eigenvalues/eigenvectors
        if s == 0:
            As = A[s].tocsr()
        else:
            As = A[s].tocsr().dot(As)
            
    x = np.ones(shape=(n_lat*n_lon,1))
    
    for i in range(100):
        x = A0.dot(x)    
        print(np.sum(x,0))

    U,s,VT = sp.linalg.svds(A0.transpose(),k=10)    
    w,u = sp.linalg.eigs(A0.transpose(),k=10)    
    
    A0 = A[0].tocsr()
    x = np.ones(shape=(n_lat*n_lon,1))/float(n_lat*n_lon)
    for i in range(10000):
        x = A0.transpose().dot(x)
    
    all_cells = range(n_lat*n_lon)
    all_lat = np.divide(all_cells,n_lon)
    all_lon = np.remainder(all_cells,n_lon)
    H = np.zeros(shape=(n_lon,n_lat))
    H[all_lon,all_lat] = x[:,0]
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131)
    ax.set_title('pcolormesh: exact bin edges')
    X, Y = np.meshgrid(lon_bins, lat_bins)
    plt.imshow(np.log(H.transpose()))
    #ax.set_aspect('equal')
    ax.show()
    
    # Write data to file
    for s in range(6):
        f = open('A%i_1_5000.txt' %s, 'w')
        mmwrite(f,A[s])
        f.close()
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131)
                    
    Extracting triple from coo matrix:
    a = A.data
    i = A.row
    j = A.col
    
    # Determine a natural spatial range for test case
    lat_min = 0
    lat_max = 0
    lon_min = 0
    lon_max = 0
    count = 0
   
    for idx in lookup.index:
        print(count)
        count += 1
        buoy = Buoy(idx, lookup, data_dir)
        tlat_min, tlat_max, tlon_min, tlon_max = buoy.spatial_range()
        lat_min = min(tlat_min,lat_min)
        lat_max = max(tlat_max,lat_max)
        lon_min = min(tlon_min,lon_min)
        lon_max = max(tlon_max,lon_max)
        del buoy
    mesh = Mesh(spatial_range,n_lat=n_lat,n_lon=n_lon)
    
    # Specify date range
    date_range = [pd.datetime(1980,03,1),pd.datetime(1990,12,1)]
    dt = pd.Timedelta('60 days')
    
    seasons = {'period':'year', 'num_seasons':6}
    
    #A = Transit(mesh, lookup, data_dir, dt,
    #                date_range=date_range, seasons=seasons)

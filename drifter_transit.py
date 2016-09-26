# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:45:58 2016

@author: hans-werner
"""

from datetime import datetime
import pandas as pd
import numpy as np
import scipy.sparse as sp
#from sklearn.preprocessing import normalize
from scipy.io import mmwrite, mmread
import matplotlib.pyplot as plt


def buoy_date_parser(month_col, day_hour_col, year_col):
    """
    Function to convert dates from data files to datetime format
    """
    month = int(month_col)
    year = int(year_col)
    day_hour = float(day_hour_col)
    day = int(day_hour)
    hour = int((day_hour - day)*24)
    return datetime(year,month,day,hour)


def load_lookup_table(data_dir, file_name):
    """
    Load the lookup file into a Dataframe
    """
    lookup_file = str('%s%s' % (data_dir,file_name))
    headers = ['file_name', 'first_line', 'last_line', 'birth_date', 'death_date']
    lookup = pd.read_table(lookup_file, sep=',', \
                           names = headers,\
                           parse_dates=True,\
                           index_col=0)
    lookup['birth_date'] = pd.to_datetime(lookup['birth_date'])
    lookup['death_date'] = pd.to_datetime(lookup['death_date'])   
    
    return lookup

    
def load_buoy_data(data_dir, file_name):
    """
    Load a drifter data file into pandas dataframe
    
    Inputs: 
    
        datadir: str, directory in which file is stored
        
        filename: str, name of buoy data file. 
        
    Outputs:
    
        data: pandas dataframe
    """  
    headers = ['date','id','lat','lon','temp','ve','vn',
                   'spd','var_lat','var_lon','var_temp']
    file_path = str('%s%s' % (data_dir,file_name))
    data = pd.read_table(file_path, delim_whitespace=True, \
                          parse_dates=[[1,2,3]], \
                          date_parser=buoy_date_parser,\
                          header=None)
    data.columns = headers 
    
    return data


def preprocess_missing_data(data):
    """
    Set 999 values to nans (panda's can deal with this better)
    """
    data.loc[data['lat'] > 990, 'lat'] = float('NaN')
    data.loc[data['lon'] > 990, 'lon'] = float('NaN')
    data.loc[data['temp'] > 990, 'temp'] = float('NaN')
    data.loc[data['ve'] > 990, 've'] = float('NaN')
    data.loc[data['vn'] > 990, 'vn'] = float('NaN')
    data.loc[data['spd'] > 990, 'spd'] = float('NaN')
    data.loc[data['var_lat'] > 990, 'var_lat'] = float('NaN')
    data.loc[data['var_lon'] > 990, 'var_lon'] = float('NaN')
    data.loc[data['var_temp'] > 990, 'var_temp'] = float('NaN')
    
    return data
    

def preprocess_fix_lon(data):
    """
    Transform longitude values in [180, 360], to [-180,0]   
    """    
    msk = (data['lon'] > 180.0)
    data.loc[msk,'lon'] = data.loc[msk].lon-360    
    return data 
    
    
def get_spatial_range(data_dir, file_list):
    """
    Construct longitudinal/latitudinal bins of a given width
    
    Inputs: 
    
        data_dir: str, path to directory that contains the data
        
        file_list: str, list of file names to consider
        
        
    Outputs: 
    
        spatial_range: {'lon_min':lon_min, 'lon_max':lon_max, 
                        'lat_min':lat_min, 'lat_max':lat_max}
    """
    #
    # Determine the spatial range
    #     
    lon_min = 0.
    lon_max = 0.
    lat_min = 0.
    lat_max = 0.
    for file_name in file_list:
        print('File: %s' % (file_name))
        data = load_buoy_data(data_dir, file_name)
        data = preprocess_missing_data(data)
        data = preprocess_fix_lon(data)
        #
        # Update Extrema
        # 
        lon_min = min(data.lon.min(), lon_min)
        lon_max = max(data.lon.max(), lon_max)
        lat_min = min(data.lat.min(), lat_min)
        lat_max = max(data.lat.max(), lat_max)
        
    spatial_range = {'lon_min':lon_min, 'lon_max':lon_max, 
                     'lat_min':lat_min, 'lat_max':lat_max}
    
    return spatial_range
    
    
def combine_transit_matrices(A,B,saveas=None):
    """
    Inputs:
        
        A, B: two sparse matrices in coo format of the same shape
        
        saveas: filename, to save resulting matrix to file instead of
            returning it. 
    """
    #
    # Check whether matrices are coo sparse
    # 
    isAcoo = sp.isspmatrix_coo(A)
    isBcoo = sp.isspmatrix_coo(B)
    if not all([isAcoo,isBcoo]):
        raise Exception('Matrices should be sparse in coo format.')
        
    #
    # Check whether matrices have the same shape
    # 
    if A.shape != B.shape:
        raise Exception('Matrices should have the same shape.')
    
    #
    # Combine matrices A and B
    # 
    cdata = np.concatenate((A.data, B.data))
    crow = np.concatenate((A.row, B.row))
    ccol = np.concatenate((A.col, B.col))
    C = sp.coo_matrix((cdata,(crow,ccol)), shape= A.shape)
    
    #
    # Save/return matrix C
    # 
    if saveas == None:
        return C
    else:
        f = open(saveas,'w')
        mmwrite(f, C)
        f.close()
    
    
def buoy_transitions(group, dt):
    """
    
    """
    ed = group.date.max()
    sd = group.date.min()
    i = group.loc[group['date'] <= ed-dt].cell.values
    j = group.loc[group['date'] >= sd+dt].cell.values
    group.loc[group['date'] <= ed-dt, 't_from'] = i
    group.loc[group['date'] <= ed-dt, 't_to'] = j
    group.loc[group['date'] >= sd+dt]['t_from'] = -1
    group.loc[group['date'] >= sd+dt]['t_to'] = -1
    return group


def build_transit_matrix(data_dir, file_name, spatial_range, dx, 
                         date_range, n_seasons, dt):
    """
    Construct transit matrix
    
    Inputs: 
        
        data_dir: str, name of the directory that contains the data files
        
        file_name: str, name of data file used

        spatial_range: dictionary with lowest and highest lon & latitudes 
            [lat_min, lat_max, lon_min, lon_max]
        
        date_range: datetime, [date_min, date_min] first & last date.
        
        n_seasons: int, should divide into 12
        
        dt: timedelte, time interval. 
    
    Outputs:
    
    
    """
        
    # =========================================================================
    # Load Data File
    # =========================================================================
    print('Loading data file')
    data = load_buoy_data(data_dir, file_name)

    # =========================================================================
    # Restrict to date range
    # =========================================================================  
    start_date, end_date = date_range
    
    date_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    if not any(date_mask):
        print 'Data falls outside date range. Exiting.'
        return 
    
    data = data.loc[date_mask]

    # =========================================================================
    # Pre-process
    # =========================================================================
    print('Set 999 to NaN')
    data = preprocess_missing_data(data)
  
    
    print('Check longitude')
    data = preprocess_fix_lon(data)
    
    # =========================================================================
    # Assign cells to positions
    # =========================================================================
    print('Assigning cell numbers to positions')
    #
    # Define mesh first
    #
    lon_min = spatial_range['lon_min']
    lon_max = spatial_range['lon_max']
    lat_min = spatial_range['lat_min']
    lat_max = spatial_range['lat_max']
    
    #
    # dx degree boxes
    #
    lon_bins = np.arange(np.floor(lon_min), np.ceil(lon_max)+dx,dx)
    lat_bins = np.arange(np.floor(lat_min), np.ceil(lat_max)+dx,dx)
    n_lon = len(lon_bins)-1
    n_lat = len(lat_bins)-1

    #
    # Bin
    #
    i_lat = pd.cut(data.lat, lat_bins, include_lowest=True, labels=False)
    i_lon = pd.cut(data.lon, lon_bins, include_lowest=True, labels=False)

    #
    # Assign cells to buoys
    #
    data['cell'] = n_lon*i_lat + i_lon


    # =========================================================================
    # Assign seasons to dates
    # =========================================================================
    print('Assigning seasons')
    #
    # Two monthly seasons
    #
    season_duration = 12/n_seasons
    season_bins = np.arange(0,13,season_duration)
    m = pd.DatetimeIndex(data['date']).month
    data['season'] = pd.cut(m, season_bins, right=True, labels=False)


    # =========================================================================
    # Count transitions
    # =========================================================================
    print('Counting transitions')
    
    
    gdata = data.groupby('id')
    f = lambda group: buoy_transitions(group,dt)
    data = gdata.apply(f)
    data['t_from'] = data.t_from.fillna(-1)
    data['t_to'] = data.t_to.fillna(-1)
    data['transitions'] = zip(data.t_from.astype(int), data.t_to.astype(int))    
    #data['transitions'] = data[['t_from','t_to']].apply(tuple, axis=1) #zip(data.t_from, data.t_to)
    #data.loc[data['t_from'].isnull(),'transitions'] = float('NaN')


    print('Forming transit matrix')
    count = []
    A = []
    for s in range(n_seasons):
        transitions_per_season = data.loc[data['season']==s,'transitions']
        count.append(transitions_per_season.value_counts(sort=False))
        i = np.array([ij[0] for ij in count[s].index.values])
        j = np.array([ij[1] for ij in count[s].index.values])
        a = np.array([c for c in count[s].astype(float)])
    
        #
        # Throw away -1 entries
        #
        ichuck = (i==-1) | (j==-1)
        i = i[~ichuck]
        j = j[~ichuck]
        a = a[~ichuck] 
        A.append(sp.coo_matrix((a,(i,j)),shape=((n_lat*n_lon),(n_lat*n_lon))))
        #A[s] = normalize(A[s], norm='l1', axis=1, copy=True)

    return A

    
if __name__ == '__main__':
    data_dir = '/home/hans-werner/Dropbox/work/projects/drifters/data/buoydata/'
    file_1 = 'buoydata_1_5000.dat'
    file_2 = 'buoydata_5001_10000.dat'
    file_3 = 'buoydata_10001_dec15.dat'
    file_list = [file_1,file_2,file_3]
    #spatial_range = get_spatial_range(data_dir, file_list)
    spatial_range = {'lon_min': -180., 'lon_max': 180., 
                     'lat_min': -78., 'lat_max': 90.}
    cellwidth = 1.
    start_date = datetime(1979,2,15,0)
    end_date = datetime(1989,12,31,23,59)
    date_range = [start_date, end_date]
    dt = pd.Timedelta('30 days')
    
    # =========================================================================
    # Experiment 1: Seasons
    # =========================================================================    
    #
    # Inputs
    #     
    lookup_file = 'lookup_table_datetime.dat'
    lookup = load_lookup_table(data_dir, lookup_file)
    cellwidth = 1.
    start_date = lookup.birth_date.min().to_datetime()
    end_date = lookup.death_date.max().to_datetime()
    date_range  = [start_date, end_date]
    dt = pd.Timedelta('30 days')
    n_season_list = [12,6,4]
    file_list = [file_1, file_2, file_3]
    
    #
    # Form matrices
    #
    A = [[],[],[]]
    season_counter = 0 
    for n_seasons in n_season_list:
        file_counter = 0
        for file_name in file_list:
            print 'File:', file_name
            Atmp = build_transit_matrix(data_dir, file_name, spatial_range, 
                                        cellwidth, date_range, n_seasons, dt)
            if file_counter == 0:
                A[season_counter] = Atmp
            else:
                for i in range(n_seasons):
                    A[season_counter][i] = \
                        combine_transit_matrices(A[season_counter][i],Atmp[i])
        season_counter += 1
    
    #
    # Combine them into yearly matrices
    # 
    Ay = [[],[],[]]
    sc = 0
    for n_seasons in n_season_list:
         for i in range(n_seasons):
             if i == 0:
                 Ay[sc] = A[sc][i].tocsr()
             else:
                 Ay[sc] = Ay[sc].dot(Ay[sc].tocsr())
    sc += 1
    
    
    # =========================================================================
    # Experiment 2: Time depence
    # =========================================================================
    

    
    
    #combine_transit_matrices(A[0], A[1],saveas='new/A12.mtx')
    
    """
    # =============================================================================
    # Tests
    # =============================================================================
    #
    # Normalize by row sums
    #  
    A0 = A[0].tocsr()
    row_sums = np.array(A0.sum(axis=1))[:,0]
    row_indices, col_indices = A0.nonzero()
    A0 /= row_sums[row_indices]
    
    #
    # Extract states with nonzero rows
    # 
    
    #
    # Compare dominant eigenvectors of 6 seasons.
    # 
    
    
    for s in range(6):
        #
        # Compute dominant eigenvalues/eigenvectors
        #
        if s == 0:
            As = A[s].tocsr()
        else:
            As = A[s].tocsr().dot(As)
            
    x = np.ones(shape=(n_lat*n_lon,1))
    print  
    for i in range(100):
        x = A0.dot(x)    
        print np.sum(x,0)
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
    # =============================================================================
    # Write data to file
    # =============================================================================
   
    for s in range(6):
        f = open('A%i_1_5000.txt' %s, 'w')
        mmwrite(f,A[s])
        f.close()
    """
    """
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131)
    
    """
    
                    
    """
    Extracting triple from coo matrix:
    a = A.data
    i = A.row
    j = A.col
    """    
    
    """
    #
    # Determine a natural spatial range for test case
    #
    lat_min = 0
    lat_max = 0
    lon_min = 0
    lon_max = 0
    count = 0
    for idx in lookup.index:
        print count
        count += 1
        buoy = Buoy(idx, lookup, data_dir)
        tlat_min, tlat_max, tlon_min, tlon_max = buoy.spatial_range()
        lat_min = min(tlat_min,lat_min)
        lat_max = max(tlat_max,lat_max)
        lon_min = min(tlon_min,lon_min)
        lon_max = max(tlon_max,lon_max)
        del buoy
    mesh = Mesh(spatial_range,n_lat=n_lat,n_lon=n_lon)
    
    #
    # Specify date range
    #
    date_range = [pd.datetime(1980,03,1),pd.datetime(1990,12,1)]
    dt = pd.Timedelta('60 days')
    
    seasons = {'period':'year', 'num_seasons':6}
    
    #A = Transit(mesh, lookup, data_dir, dt,
    #                date_range=date_range, seasons=seasons)
    """
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 16:45:58 2016

@author: hans-werner
"""


from datetime import datetime
from scipy.io import mmwrite, mmread
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize


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
    # Determine the spatial range
    lon_min = 0.
    lon_max = 0.
    lat_min = 0.
    lat_max = 0.
    for file_name in file_list:
        print('File: %s' % (file_name))
        data = load_buoy_data(data_dir, file_name)
        data = preprocess_missing_data(data)
        data = preprocess_fix_lon(data)
        
        # Update Extrema
        lon_min = min(data.lon.min(), lon_min)
        lon_max = max(data.lon.max(), lon_max)
        lat_min = min(data.lat.min(), lat_min)
        lat_max = max(data.lat.max(), lat_max)
        
    spatial_range = {'lon_min':lon_min, 'lon_max':lon_max, 
                     'lat_min':lat_min, 'lat_max':lat_max}
    return spatial_range
    
    
def combine_transit_matrices(A, B, saveas=None):
    """
    Inputs:
        
        A, B: two sparse matrices in coo format of the same shape
        
        saveas: filename, to save resulting matrix to file instead of
            returning it. 
    """
    # Check whether matrices are coo sparse
    isAcoo = sp.isspmatrix_coo(A)
    isBcoo = sp.isspmatrix_coo(B)
    if not all([isAcoo,isBcoo]):
        raise Exception('Matrices should be sparse in coo format.')
        
    # Check whether matrices have the same shape
    if A.shape != B.shape:
        raise Exception('Matrices should have the same shape.')
    
    # Combine matrices A and B
    cdata = np.concatenate((A.data, B.data))
    crow = np.concatenate((A.row, B.row))
    ccol = np.concatenate((A.col, B.col))
    C = sp.coo_matrix((cdata,(crow,ccol)), shape= A.shape)
    
    # Save/return matrix C
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


def neighbors(matrix, i, j):
    """
    neighbors returns the positions of the neighbors of cell (i,j)
    in the given matrix.

    These are always in the domain of the matrix and guaranteed
    to be valid positions.
    """
    directions = [
        # Verticals and horizontals.
        (0,  -1), (0,   1), (1,  0), (-1, 0),
        # Diagonals.
        (1,  -1), (-1, -1), (-1, 1), (1,  1)
    ]
    neighbors = []
    for direction in directions:
        # Check if horizontal directions are valid.
        if i + direction[0] >= len(matrix) or i + direction[0] < 0:
            continue
        # Check if vertical directions are valid.
        if j + direction[1] >= len(matrix) or j + direction[1] < 0:
            continue
        # If both horizontal and vertical directions are valid, add
        # this neighbor position.
        neighbors.append((i + direction[0], j + direction[1]))
    return neighbors


def fix_dangling_node(transition_matrix, i, j):
    ij_neighbors = neighbors(transition_matrix, i, j)

    for neighbor in ij_neighbors:
        transition_matrix[neighbor[0]][neighbor[1]] = 1/len(ij_neighbors)


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
    data = load_buoy_data(data_dir, file_name)

    # =========================================================================
    # Restrict to date range
    # =========================================================================  
    start_date, end_date = date_range
    
    date_mask = (data['date'] >= start_date) & (data['date'] <= end_date)
    if not any(date_mask):
        print('Data falls outside date range. Exiting.')
        return 
    
    data = data.loc[date_mask]

    # =========================================================================
    # Pre-process
    # =========================================================================
    
    # Set data <= 999 to NaN
    data = preprocess_missing_data(data)
  
    # Check longitude
    data = preprocess_fix_lon(data)
    
    # =========================================================================
    # Assign cells to positions
    # =========================================================================
    
    # Define mesh first
    lon_min = spatial_range['lon_min']
    lon_max = spatial_range['lon_max']
    lat_min = spatial_range['lat_min']
    lat_max = spatial_range['lat_max']
    
    # dx degree boxes
    lon_bins = np.arange(np.floor(lon_min), np.ceil(lon_max)+dx,dx)
    lat_bins = np.arange(np.floor(lat_min), np.ceil(lat_max)+dx,dx)
    n_lon = len(lon_bins)-1
    n_lat = len(lat_bins)-1

    # Bin
    i_lat = pd.cut(data.lat, lat_bins, include_lowest=True, labels=False)
    i_lon = pd.cut(data.lon, lon_bins, include_lowest=True, labels=False)

    # Assign cells to buoys
    data['cell'] = n_lon*i_lat + i_lon

    # =========================================================================
    # Assign seasons to dates
    # =========================================================================
    
    # Two monthly seasons
    season_duration = 12/n_seasons
    season_bins = np.arange(0,13,season_duration)
    m = pd.DatetimeIndex(data['date']).month
    data['season'] = pd.cut(m, season_bins, right=True, labels=False)

    # =========================================================================
    # Count transitions
    # =========================================================================
    
    gdata = data.groupby('id')
    f = lambda group: buoy_transitions(group,dt)
    data = gdata.apply(f)
    data['t_from'] = data.t_from.fillna(-1)
    data['t_to'] = data.t_to.fillna(-1)
    data['transitions'] = list(zip(data.t_from.astype(int), data.t_to.astype(int))) 
    #data['transitions'] = data[['t_from','t_to']].apply(tuple, axis=1) #zip(data.t_from, data.t_to)
    #data.loc[data['t_from'].isnull(),'transitions'] = float('NaN')

    # =========================================================================
    # Forming transit matrix
    # =========================================================================
    count = []
    A = []
    for s in range(n_seasons):
        transitions_per_season = data.loc[data['season']==s,'transitions']
        count.append(transitions_per_season.value_counts(sort=False))
        i = np.array([ij[0] for ij in count[s].index.values])
        j = np.array([ij[1] for ij in count[s].index.values])
        a = np.array([c for c in count[s].astype(float)])
    
        # Throw away -1 entries
        ichuck = (i==-1) | (j==-1)
        i = i[~ichuck]
        j = j[~ichuck]
        a = a[~ichuck] 
        A.append(sp.coo_matrix((a,(i,j)),shape=((n_lat*n_lon),(n_lat*n_lon))))

    return A

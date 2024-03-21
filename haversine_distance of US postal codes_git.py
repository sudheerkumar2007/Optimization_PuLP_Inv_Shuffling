# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:59:51 2022

@author: user
"""

#Find distance by postal codes
import pandas as pd
import numpy as np
from uszipcode import SearchEngine
from sklearn.neighbors import DistanceMetric

#import pgeocode
#dist = pgeocode.GeoDistance('US')
#dist.query_postal_code(14750, 22033)
#dist.query_postal_code(77566,80920)*0.62 # distance between store 63 and 67

#Method1
def get_coordinates(zip_code):
    search = SearchEngine()
    zip1 = search.by_zipcode(zip_code)
    lat1 =zip1.lat
    long1 =zip1.lng
    return lat1,long1

def sklearn_haversine(lat, lon):
    haversine = DistanceMetric.get_metric('haversine')
    latlon = np.hstack((lat[:, np.newaxis], lon[:, np.newaxis]))
    dists = haversine.pairwise(latlon)
    return 3956*dists # Radius of earth in Miles

zips = pd.read_csv("Path to file")
zips[['PIN1','PIN2']] = zips['Postal_Code'].str.split('-', 2,expand= True)
zips = zips.drop(columns = ['PIN2','Postal_Code']).rename(columns = {'PIN1':'Postal_Code'})
zips['Latitude'], zips['Longitude'] = zip(*zips['Postal_Code'].apply(get_coordinates))
new_zip = zips[['Postal_Code', 'Latitude', 'Longitude']].set_index('Postal_Code')
lat=new_zip['Latitude'].to_numpy()
lon=new_zip['Longitude'].to_numpy()
d_m = sklearn_haversine(lat, lon)


#Method2 - Preferred
import mpu
lat1=new_zip['Latitude'].to_numpy()
lon1=new_zip['Longitude'].to_numpy()

# create a matrix for the distances between each pair of zones
distances = np.zeros((len(zips), len(zips)))

for i in range(len(zips)):
    for j in range(len(zips)):
#        distances[i, j] = haversine(zips.iloc[i], zips.iloc[j])
        distances[i,j] = mpu.haversine_distance((lat1[i],lon1[i]),(lat1[j],lon1[j]))*0.62

mat = pd.DataFrame(distances, index=zips.index, columns=zips.index)
mat.columns = zips['STORE_NUM']
mat.index = zips['STORE_NUM']
mat=mat.reset_index()
mat.to_csv("Path of dir")

#From internet - Method 3 - Throwing error
def haversine(first, second):
    # convert decimal degrees to radians
    lat, lon, lat2, lon2 = map(np.radians, [first[0], first[1], second[0], second[1]])

    # haversine formula
    dlon = lon2 - lon
    dlat = lat2 - lat
    a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
#    r_km = 6371 # Radius of earth in kilometers
    r_miles = 3956 # Radius of earth in Miles
    return c * r_miles

distances1 = np.zeros((len(zips), len(zips)))
for i in range(len(zips)):
    for j in range(len(zips)):
        distances1[i, j] = haversine(zips.iloc[i], zips.iloc[j])

mat1 = pd.DataFrame(distances, index=zips.index, columns=zips.index)


#Method4
#Not haversine, But general distance between 2 points
from scipy.spatial import distance_matrix
distance_matrix()

dist_mat = pd.DataFrame(distance_matrix(new_zip.values, new_zip.values), index=new_zip.index, columns=new_zip.index)











zip2 =search.by_zipcode('80920')
lat2 =zip2.lat
long2 =zip2.lng




dist =round(mpu.haversine_distance(zip_00501,zip_00544),2)
print(dist)
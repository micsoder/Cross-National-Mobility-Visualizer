import pandas as pd
import geopandas as gpd
import numpy as np
import math


class BwHandler():

    def __init__(self):
        self.df = self.__csv_to_df()
        self.unique_lst = self.save_cntr_od_to_lst()
        self.bw_data_handling_initialized()

    def __csv_to_df(self):
            
        return pd.read_csv('mobility_data.csv', sep = ',')
    
    def save_cntr_od_to_lst(self):
        self.df = self.df.dropna()

        lst_of_unique_cntr_od = self.df['CNTR_OD'].unique().tolist()
        print(lst_of_unique_cntr_od)

        return lst_of_unique_cntr_od
    
    def bw_data_handling_initialized(self):
        bandwidth_df = pd.DataFrame(columns=['country_pair', 'country', 'search_radius', 'cntr_coef', 'BW'])
        self.unique = ''
        for unique_od in self.unique_lst:
            print(unique_od)
            self.unique = unique_od
            self.unique_df = self.df.loc[self.df['CNTR_OD'].isin([unique_od])]

            unique_cntr_1 = self.unique[:2]
            print(unique_cntr_1)
            unique_cntr_2 = self.unique[3:5]
            print(unique_cntr_2)

            self.country_pair_gdf = self.create_country_pair_gdf(self.unique_df)
            self.country_1_coordinates =  self.country_coords_gdf(unique_cntr_1)
            self.country_2_coordinates = self.country_coords_gdf(unique_cntr_2)

            parameter_df = pd.DataFrame(columns=['country_pair', 'country', 'N', 'Dm', 'SD', 'search_radius'])
            country1_bandwidth_df = self.calculate_parameters_for_search_radius(self.country_1_coordinates, parameter_df)
            country2_bandwidth_df = self.calculate_parameters_for_search_radius(self.country_2_coordinates, parameter_df)
            bandwidth = self.calculate_bandwidth(self.country_1_coordinates, self.country_2_coordinates, country1_bandwidth_df, country2_bandwidth_df, bandwidth_df)
        

        print("All done and calculated")
        print(len(bandwidth))
        print(bandwidth.head())
        print(bandwidth.tail())

      

    def create_country_pair_gdf(self, unique_df):

        gdf_start = self.df_to_gdf(self.unique_df.h3_grid_res10_start_lon, self.unique_df.h3_grid_res10_start_lat)
        gdf_end = self.df_to_gdf(self.unique_df.h3_grid_res10_end_lon, self.unique_df.h3_grid_res10_end_lat)

        country_pair_gdf = self.combine_gdfs(gdf_start, gdf_end)

        return country_pair_gdf


    def df_to_gdf(self, lon, lat):

        gdf = gpd.GeoDataFrame(
        self.unique_df, geometry = gpd.points_from_xy(lon, lat))
        gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(epsg=3035)

        return gdf
        

    def combine_gdfs(self, gdf_start, gdf_end):

        country_pair_gdf = gdf_start.set_geometry('geometry').rename(columns={'geometry':'geometry_of_start'})
        country_pair_gdf['geometry_of_end'] = gdf_end['geometry']
        
        return country_pair_gdf


    def country_coords_gdf(self, country):

        filtered_start_gdf = self.country_pair_gdf.loc[self.country_pair_gdf['CNTR_ID_start'].isin([country])]
        filtered_end_gdf = self.country_pair_gdf.loc[self.country_pair_gdf['CNTR_ID_end'].isin([country])]


        if not filtered_start_gdf.empty:
            filtered_start_gdf = filtered_start_gdf.drop(columns=['geometry_of_end'])
            filtered_start_gdf = filtered_start_gdf.rename(columns={'geometry_of_start': 'geometry'})

        if not filtered_end_gdf.empty:
            filtered_end_gdf = filtered_end_gdf.drop(columns=['geometry_of_start'])
            filtered_end_gdf = filtered_end_gdf.rename(columns={'geometry_of_end': 'geometry'})
        
        country_gdf = pd.concat([filtered_start_gdf, filtered_end_gdf])
        country_gdf = country_gdf.dropna()
        country_id = str(country)

        country_gdf['country_name'] = country_id
        
        country_gdf = country_gdf.reset_index(drop=True)
        print(country_gdf.head())
        return country_gdf


    def calculate_parameters_for_search_radius(self, country, parameter_df): 

        # Calculate the count of all points
        self.N = len(country)

        # Calculate the mean center (centroid) of all input points
        centroid = country.geometry.centroid
        MeanCenter = (centroid.x.mean(), centroid.y.mean())
            
        # Calculate the distance from the Mean Center for all points
        DistMC = country.distance(gpd.points_from_xy([MeanCenter[0]], [MeanCenter[1]])[0])
        
        # Calculate the median of these distances (DistMC)
        self.Dm = np.median(DistMC)

        # Calculate the sum of squared deviations
        sq_dev = np.sum((DistMC - self.Dm) ** 2)

        # Calculate the Standard Distance
        self.SD = np.sqrt(sq_dev / (self.N - 1))
        

        print(f'The number of points: {self.N}')
        print(f'The median of distance: {self.Dm}')
        print(f'The Standard Distance: {self.SD}')

        self.calculate_search_radius(self.Dm, self.SD, self.N, country, parameter_df)
        return parameter_df

    def calculate_search_radius(self, Dm, SD, N, country, parameter_df):

        country_name = country.iloc[0]['country_name']

        ln2 = math.log(2)
        min_value = min(SD, math.sqrt(1/ln2 * Dm))
        search_radius = 0.9 * min_value * N **(-0.2)
        print(f'country: {country_name}, search radius : {search_radius}')

        parameter_df.loc[len(parameter_df)] = [self.unique, country_name, N, Dm, SD, search_radius]

        return parameter_df

    
    def calculate_bandwidth(self, country1, country2, country1_parameter, country2_parameter, bandwidth_df):
    
        cntr1_search_radius = country1_parameter['search_radius'][0]
        cntr2_search_radius = country2_parameter['search_radius'][1]


        if cntr1_search_radius > cntr2_search_radius:

            coef_cntr_1 = cntr1_search_radius/cntr1_search_radius
            coef_cntr_2 = cntr1_search_radius/cntr2_search_radius
            cntr_1_bw = 0.5
            cntr_2_bw = coef_cntr_2/coef_cntr_1*cntr_1_bw

            self.save_bw_to_df(country1, country1_parameter, cntr1_search_radius, coef_cntr_1, cntr_1_bw, bandwidth_df)
            self.save_bw_to_df(country2, country2_parameter, cntr2_search_radius, coef_cntr_2, cntr_2_bw, bandwidth_df)
        
        if cntr1_search_radius < cntr2_search_radius:

            coef_cntr_2 = cntr2_search_radius/cntr2_search_radius
            coef_cntr_1 = cntr2_search_radius/cntr1_search_radius
            cntr_2_bw = 0.5
            cntr_1_bw = coef_cntr_1/coef_cntr_2*cntr_2_bw

            self.save_bw_to_df(country1, country1_parameter, cntr1_search_radius, coef_cntr_1, cntr_1_bw, bandwidth_df)
            self.save_bw_to_df(country2, country2_parameter, cntr2_search_radius, coef_cntr_2, cntr_2_bw, bandwidth_df)
    
    def save_bw_to_df(self, country, country_parameter, search_radius, cntr_coef, cntr_bw, bandwidth_df):

        country_name = country.iloc[0]['country_name']


        bandwidth_df.loc[len(bandwidth_df)] = [self.unique, country_name, search_radius, cntr_coef, cntr_bw]
        bandwidth_df.round()

        return bandwidth_df

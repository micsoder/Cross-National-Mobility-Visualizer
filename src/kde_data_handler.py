import pandas as pd 
import geopandas as gpd
import seaborn as sns
import contextily
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.path as mpath
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box
from descartes import PolygonPatch
from shapely.geometry import MultiPolygon
from scipy.interpolate import griddata
from scipy.spatial import distance
import math
#from matplotlib.patches import Polygon as PolygonPatch


class KdeDataHandler():
    
    def __init__(self, countries):
        self.countries = countries
        self.df = self.__csv_to_df() 
        self.cntr_id_country = self.__get_cntr_id(self.countries)
        self.cntr_od = self.__get_cntr_od(self.countries)


    def start_processing(self):
        
        print("Data processing starting...") 
        print(' ')      
        self.country_pair_df = self.__create_df_of_cntr_od()
        self.country_pair_gdf = self.__create_country_pair_gdf()
        self.country_1_coordinates = self.__country_coords_gdf(self.countries[0])
        self.country_2_coordinates = self.__country_coords_gdf(self.countries[1])


        self.gpkg_file = self.read_gpkg_file()
        self.interreg_file = self.importing_interreg()
        print('Search radius calculation started')
        #self.search_radius = self.bandwidth_calculation_initialized()
        print("Data processing done...")
        print(' ')


    def visualize(self):
        print("Visualization starting...")
        print(' ')
        #self.kde_plot_initializing_for_interreg()
        #self.kde_plot_initializing_for_entire_dataset()
        print(' ')
        print('Visualization done.')

    # Reading in the csv file
    def __csv_to_df(self):
        
        #return pd.read_csv('full_mobility_dataset.csv', sep = ',')
        return pd.read_csv('full_mobility_dataset_without_int.csv', sep = ',')
        
    def __get_cntr_id(self, countries):
        country1 = countries[0]
        country2 = countries[1]

        return country1, country2

    def __get_cntr_od(self, countries):

        country_pair1 = f"{countries[0]}_{countries[1]}"
        country_pair2 = f"{countries[1]}_{countries[0]}" #avoiding error in case of wrong order

        return self.__sanatize_cntr_od(country_pair1, country_pair2)

    def __sanatize_cntr_od(self, country_pair1, country_pair2):

        if self.df['CNTR_OD'].str.contains(country_pair1).any():
            return country_pair1
        
        if self.df['CNTR_OD'].str.contains(country_pair2).any():
            return country_pair2

    def __create_df_of_cntr_od(self):

        return self.df.loc[self.df['CNTR_OD'].isin([self.cntr_od])]
    
        
    def __create_country_pair_gdf(self):

        gdf_start = self.__df_to_gdf(self.country_pair_df.start_lon, self.country_pair_df.start_lat)
        gdf_end = self.__df_to_gdf(self.country_pair_df.end_lon, self.country_pair_df.end_lat)

        country_pair_gdf = self.__combine_gdfs(gdf_start, gdf_end)

        return country_pair_gdf

    def __df_to_gdf(self, lon, lat):

        gdf = gpd.GeoDataFrame(
        self.country_pair_df, geometry = gpd.points_from_xy(lon, lat))
        gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(epsg=3035)

        return gdf
        
    def __combine_gdfs(self, gdf_start, gdf_end):

        country_pair_gdf = gdf_start.set_geometry('geometry').rename(columns={'geometry':'geometry_of_start'})
        country_pair_gdf['geometry_of_end'] = gdf_end['geometry']
        
        return country_pair_gdf
    

    def __country_coords_gdf(self, country):

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
        return country_gdf

    
    def bandwidth_calculation_initialized(self):

        bandwidth_df = pd.DataFrame(columns=['country_pair', 'country', 'N', 'Dm', 'SD', 'search_radius'])

        #self.calculate_bandwidth(self.country_1_coordinates, bandwidth_df)
        #self.calculate_bandwidth(self.country_2_coordinates, bandwidth_df)

        self.calculate_bandwidth(self.country_1_dataset, bandwidth_df)
        self.calculate_bandwidth(self.country_2_dataset, bandwidth_df)

    def calculate_bandwidth(self, country, bandwidth_df): 

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

        self.calculate_the_exact_bandwidth(self.Dm, self.SD, self.N, country, bandwidth_df)
    
    def calculate_the_exact_bandwidth(self, Dm, SD, N, country, bandwidth_df):

        country_name = country.iloc[0]['country_name']

        ln2 = math.log(2)
        min_value = min(SD, math.sqrt(1/ln2 * Dm))
        search_radius = 0.9 * min_value * N **(-0.2)
        print(f'country: {country_name}, search radius : {search_radius}')

        bandwidth_df.loc[len(bandwidth_df)] = [self.cntr_od, country_name, N, Dm, SD, search_radius]

        print(bandwidth_df.head())
        bandwidth_df.to_csv('calculated_search_radius_int.csv', sep=',', index = False)


    def contour_intervalls(self, number_of_intervalls):
        first_intervall_value = 0.05
        actual_intervalls = number_of_intervalls - 1
        intervall = 1/number_of_intervalls
        value = 0 
        levels_list = []
        levels_list.append(first_intervall_value)
        for i in range(actual_intervalls):
            value += intervall
            levels_list.append(value)
        return levels_list


    def read_gpkg_file(self):

        #self.border_data = gpd.read_file('GRL_region.gpkg')
        self.border_data = gpd.read_file('NUTS_test.gpkg')
        #self.border_data = gpd.read_file('cropped_greatLux.gpkg')

        self.border_data = self.border_data.to_crs(epsg = 3035)


    def importing_interreg(self):
        self.interreg_data = gpd.read_file('NUTS_3_PT__ES2_without_fid.gpkg')
        self.interreg_data = self.interreg_data.to_crs(epsg = 3035)

        self.interreg_data = self.interreg_data[(self.interreg_data['Interreg'] == 'TRUE')]

        self.int_regions1 = self.selecting_interreg(self.country_1_coordinates)
        self.int_regions2 = self.selecting_interreg(self.country_2_coordinates)

        self.clipped_data1 = self.clipping_to_interreg(self.country_1_coordinates, self.int_regions1)
        self.clipped_data1 = self.clipped_data1.reset_index(drop=True)
        self.clipped_data1['Interreg_ES'] = True
        

        self.clipped_data2 = self.clipping_to_interreg(self.country_2_coordinates, self.int_regions2)
        self.clipped_data2 = self.clipped_data2.reset_index(drop=True)
        self.clipped_data2['Interreg_PT'] = True
        

        self.merged_country1 = self.merge_clipped_interreg(self.country_1_coordinates, self.clipped_data1)
        self.merged_country2 = self.merge_clipped_interreg(self.country_2_coordinates, self.clipped_data2)

        self.filtered_merged = self.merge_countries_to_one(self.merged_country1, self.merged_country2)

        self.country1_interreg_data = self.filtered_merged.loc[:, self.filtered_merged.columns.str.endswith('_x')]

        self.country_1_dataset = self.removing_end(self.country1_interreg_data, '_x')
        
        self.country_1_dataset = self.country_1_dataset.set_geometry('geometry')
                # Save the merged GeoDataFrame to a .gpkg file
        self.country_1_dataset.to_file('ES_KDE_input_data.gpkg', driver='GPKG')


        self.country2_interreg_data = self.filtered_merged.loc[:, self.filtered_merged.columns.str.endswith('_y')]

        self.country_2_dataset = self.removing_end(self.country2_interreg_data, '_y')
        self.country_2_dataset = self.country_2_dataset.set_geometry('geometry')
        self.country_2_dataset.to_file('PT_KDE_input_data.gpkg', driver='GPKG')

        print("saved")

    def selecting_interreg(self, country):

        country_abb = country.iloc[0]['country_name']

        if country_abb == "ES":
            self.int_regions = self.interreg_data.loc[self.interreg_data['CNTR_CODE'].isin([country_abb])]
        
        elif country_abb == 'PT':
            self.int_regions = self.interreg_data.loc[self.interreg_data['CNTR_CODE'].isin([country_abb])]
        
        return self.int_regions
    
    def clipping_to_interreg(self, country, region):

        return gpd.clip(country, region)
    
    def merge_clipped_interreg(self, country, clipped_country):

        country_abb = country.iloc[0]['country_name']
        interreg_column = f'Interreg_{country_abb}'

        self.merged = pd.merge(country, clipped_country[['id', interreg_column]], on = 'id', how = 'left')
        # fill the NaN values in the new column with False
        self.merged[interreg_column] = self.merged[interreg_column].fillna(False)

        return self.merged
    
    def merge_countries_to_one(self, country1, country2):

        # merge the two GeoDataFrames on the 'id' column
        merged = country1.merge(country2, on='id')

        merged = merged.set_geometry('geometry_x')

        # create a new column 'Interreg_status' in the merged GeoDataFrame and set its values to True where both 'Interreg_ES' and 'Interreg_PT' are True
        merged['Interreg_status'] = (merged['Interreg_ES'] == True) & (merged['Interreg_PT'] == True)

        # filter the merged GeoDataFrame to only include rows where 'Interreg_status' is True
        filtered = merged[merged['Interreg_status'] == True]

        return filtered
    
    def removing_end(self, country, letter):

        country_ex = country.copy()

        # loop through all columns and remove '_x'
        for col in country_ex.columns:
            if col.endswith(letter):
                country_ex.rename(columns={col: col[:-2]}, inplace=True)

        print(country_ex.columns)
        return country_ex

    def kde_plot_initializing_for_interreg(self):

        #first country

        self.kde1 = self.kde_plot(self.country_1_dataset, 0.5)
        print("KDE plot done")
        self.country_1_file_name = self.kde_to_gpkg(self.kde1, self.country_1_dataset)
        print("KDE plot saved as gpkg")
        self.selected_regions_1 = self.select_region(self.country_1_dataset)
        print("Region selected")
        self.country_1_plot = self.read_geo_file(self.country_1_dataset, self.country_1_file_name, self.selected_regions_1)
        print("Clipped plot saved")

        #second country

        self.kde2 = self.kde_plot(self.country_2_dataset, 0.54)
        print("KDE plot done")
        self.country_2_file_name = self.kde_to_gpkg(self.kde2, self.country_2_dataset)
        print("KDE plot saved as gpkg")
        self.selected_regions_2 = self.select_region(self.country_2_dataset)
        print("Region selected")
        self.country_2_plot = self.read_geo_file(self.country_2_dataset, self.country_2_file_name, self.selected_regions_2)
        print("Clipped plot saved")


        #Merging of country 1 and country 2
        self.merge_clipped_layer(self.country_1_plot, self.country_2_plot, self.selected_regions_1, self.selected_regions_2)

    def kde_plot_initializing_for_entire_dataset(self):

        #first country
        self.kde1 = self.kde_plot(self.country_1_coordinates, 0.75)
        print("KDE plot done")
        self.country_1_file_name = self.kde_to_gpkg(self.kde1, self.country_1_coordinates)
        print("KDE plot saved as gpkg")
        self.selected_regions_1 = self.select_region(self.country_1_coordinates)
        print("Region selected")
        self.country_1_plot = self.read_geo_file(self.country_1_coordinates, self.country_1_file_name, self.selected_regions_1)
        print("Clipped plot saved")

        #second country
        self.kde2 = self.kde_plot(self.country_2_coordinates, 1.05)
        print("KDE plot done")
        self.country_2_file_name = self.kde_to_gpkg(self.kde2, self.country_2_coordinates)
        print("KDE plot saved as gpkg")
        self.selected_regions_2 = self.select_region(self.country_2_coordinates)
        print("Region selected")
        self.country_2_plot = self.read_geo_file(self.country_2_coordinates, self.country_2_file_name, self.selected_regions_2)
        print("Clipped plot saved")


        #Merging of country 1 and country 2
        self.merge_clipped_layer(self.country_1_plot, self.country_2_plot, self.selected_regions_1, self.selected_regions_2)

    def kde_plot(self, country, bw):


        levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]

        self.kde = sns.kdeplot(
            x = country.geometry.x,
            y = country.geometry.y,
            cmap = 'viridis',
            fill = True,
            alpha = 0.5,
            bw_adjust = bw,
            levels = levels
        )

        return self.kde

    def kde_to_gpkg(self, kde, country):
        #fix the levels 
        levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]

        level_polygons = []
        i = 0
        for col in kde.collections:
            paths = []
            # Loop through all polygons that have the same intensity level
            for contour in col.get_paths(): 
                # Create a polygon for the countour
                # First polygon is the main countour, the rest are holes
                for ncp,cp in enumerate(contour.to_polygons()):
                    x = cp[:,0]
                    y = cp[:,1]
                    new_shape = Polygon([(i[0], i[1]) for i in zip(x,y)])
                    if ncp == 0:
                        poly = new_shape
                    else:
                        # Remove holes, if any
                        poly = poly.difference(new_shape)

                # Append polygon to list
                paths.append(poly)
            # Create a MultiPolygon for the contour
            multi = MultiPolygon(paths)
            # Append MultiPolygon and level as tuple to list
            level_polygons.append((levels[i], multi))
            i+=1

        # Create DataFrame
        df_of_polygons = pd.DataFrame(level_polygons, columns =['level', 'geometry'])
        # Convert to a GeoDataFrame
        gdf_of_polygons = gpd.GeoDataFrame(df_of_polygons, geometry='geometry', crs = country.crs)
        print(gdf_of_polygons.crs)
        # Set CRS for geometric operations
        gdf_of_polygons = gdf_of_polygons.to_crs(epsg=3035)
        print(gdf_of_polygons.crs)
        # Calculate area
        gdf_of_polygons['area'] = gdf_of_polygons['geometry'].area
        #file name 
        country_id= country.iloc[0]['country_name']
        print(country_id)
        file_name = f'{country_id}_geo_file.gpkg'
        print(file_name)

        # Save to file
        gdf_of_polygons.to_file(file_name, driver='GPKG')
        return file_name



    def select_region(self, country):

        country_abb = country.iloc[0]['country_name']

        if country_abb == 'DE':
            self.selected_regions = self.border_data.loc[self.border_data['CNTR_CODE'].isin(['GM'])]

        else:
            self.selected_regions = self.border_data.loc[self.border_data['CNTR_CODE'].isin([country_abb])]

        self.selected_regions = self.selected_regions.reset_index(drop=True)

        self.selected_regions.set_crs(3035)
        self.selected_regions.to_crs(epsg = 3035)

        print("printing out the selected regions head")
        print(self.selected_regions.head())

        return self.selected_regions


    def read_geo_file(self, country, filename, region):

        kde_vector_layer = gpd.read_file(filename)
        print("currently printing out the kde vector layer head")
        print(kde_vector_layer.head())


        # Get the intersection of the kde_vector_layer with the selected_regions
        clipped_layer = gpd.overlay(kde_vector_layer, region, how='intersection')


        # Plot the clipped layer using the plot() method
        fig, ax = plt.subplots()
        clipped_layer.plot(ax=ax)

        # Plot the polygon on top of the clipped layer to show the clipping extent
        region.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)

        #contextily.add_basemap(ax, crs = 'EPSG:3035')

        # Show the plot
        plt.show()

        return clipped_layer
    

    def merge_clipped_layer(self, clipped_layer1, clipped_layer2, region1, region2):

        merged_layers = pd.concat([clipped_layer1, clipped_layer2], ignore_index = True)


        ax = merged_layers.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')

        region1.plot(ax=ax, alpha = 0.5, facecolor = 'white', edgecolor = 'black')
        region2.plot(ax=ax, alpha = 0.5, facecolor = 'white', edgecolor = 'black')


        # Show the plot
        contextily.add_basemap(ax, crs = 'EPSG:3035')
        plt.show()

        
        # Save the merged GeoDataFrame to a .gpkg file
        merged_layers.to_file('countries_merged_ES_PT_int.gpkg', driver='GPKG')





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
        #self.calculated_xy = self.min_max_xy_calculation()
        self.country_1_bandwidth = self.calculate_bandwidth(self.country_1_coordinates)
        self.country_2_bandwidth = self.calculate_bandwidth(self.country_2_coordinates)
        print("Data processing done...")
        print(' ')


    def visualize(self):
        print("Visualization starting...")
        print(' ')
        #self.region_viz(self.cntr_id_country[0], self.country_1_coordinates)
        #self.kde_to_gpkg(self.country_1_coordinates)
        print(' ')
        print('Visualization done.')

    # Reading in the csv file
    def __csv_to_df(self):
        
        return pd.read_csv('mobility_data.csv', sep = ',')
        
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

        gdf_start = self.__df_to_gdf(self.country_pair_df.h3_grid_res10_start_lon, self.country_pair_df.h3_grid_res10_start_lat)
        gdf_end = self.__df_to_gdf(self.country_pair_df.h3_grid_res10_end_lon, self.country_pair_df.h3_grid_res10_end_lat)

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
    
    def min_max_xy_calculation(self):

        min_max_xy_df = pd.DataFrame(columns=['country_pair','country', 'x_min', 'x_max', 'y_min', 'y_max'])

        self.calculate_xy(self.country_1_coordinates, min_max_xy_df)
        self.calculate_xy(self.country_2_coordinates, min_max_xy_df)

    def calculate_xy(self, country, min_max_xy_df):

        country_name = country.iloc[0]['country_name']

        country_bounds = country.bounds

        x_min = country_bounds['minx'].min()
        x_max = country_bounds['maxx'].max()
        y_min = country_bounds['miny'].min()
        y_max = country_bounds['maxy'].max()

        min_max_xy_df.loc[len(min_max_xy_df)] = [self.cntr_od, country_name, x_min, x_max, y_min, y_max]

        print(min_max_xy_df.head())
        min_max_xy_df.to_csv('calculated_bounds.csv', sep=',', index = False)
    

    def calculate_bandwidth(self, country): 

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
        
        # Calculate the Standard Distance of these distances (DistMC)
        #self.std = np.std(DistMC)

        print(f'The number of points: {self.N}')
        print(f'The median of distance: {self.Dm}')
        print(f'The Standard Distance: {self.SD}')
        #print(f'Std: {self.std}')

        self.calculate_the_exact_bandwidth(self.Dm, self.SD, self.N, country)
    
    def calculate_the_exact_bandwidth(self, Dm, SD, N, country):

        country_name = country.iloc[0]['country_name']

        ln2 = math.log(2)
        min_value = min(SD, math.sqrt(1/ln2 * Dm))
        search_radius = 0.7 * min_value * N **(-0.4)
        print(f'country: {country_name}, search radius : {search_radius}')





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
        self.border_data = gpd.read_file('cropped_greatLux.gpkg')

        self.border_data = self.border_data.to_crs(epsg = 3035)


    def region_viz(self, country1_abb, country):

        if country1_abb == 'DE':
            self.selected_regions = self.border_data.loc[self.border_data['FIPS'].isin(['GM'])]
        else: 
            self.selected_regions = self.border_data.loc[self.border_data['FIPS'].isin([country1_abb])]

        self.selected_regions = self.selected_regions.reset_index(drop=True)

        ax = self.selected_regions.plot(figsize=(10, 8), alpha = 0.5, facecolor = 'white', edgecolor = 'black')

        print(f'Selected regions crs: {self.selected_regions.crs}')
        print(f'Country crs: {country.crs}')
        print(' ')
        print('Selected regions head:')
        print(self.selected_regions.head())
        print(' ')
        print('Country head:')
        print(country.head())

        country_clipped = country.clip(self.selected_regions)

        self.kde = sns.kdeplot(
            x = country_clipped.geometry.x,
            y = country_clipped.geometry.y,
            cmap = 'viridis',
            fill = True,
            alpha = 0.5,
            ax = ax,
            bw_adjust = 6,
            levels = self.contour_intervalls(4)
        )

        ax.set_xlim(self.selected_regions.total_bounds[0], self.selected_regions.total_bounds[2])
        ax.set_ylim(self.selected_regions.total_bounds[1], self.selected_regions.total_bounds[3])


        contextily.add_basemap(ax, crs = 'EPSG:3035')

        plt.show()


    def kde_to_gpkg(self, country):

        levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]

        level_polygons = []
        i = 0
        for col in self.kde.collections:
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
        # Save to file
        gdf_of_polygons.to_file('geo_file_TEST.gpkg', driver='GPKG')











    
    def first_region_viz(self, country1, country):
        
        #self.selected_regions = self.border_data.loc[self.border_data['FIPS'].isin([country1])]
        #self.lorraine = self.border_data.loc[self.border_data['OriginUnit'].isin(['Lorraine'])]
        #self.selected_regions = self.selected_regions.explode(ignore_index=True)
        #exploded = self.selected_regions.explode(ignore_index=True)
        #mask = MultiPolygon(self.selected_regions.geometry.values).buffer(0)
        levels1 = [0.2,0.4,0.6,0.8,1]
        levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]


        f, ax = plt.subplots(ncols=1, figsize=(20, 8))

        #ax = self.selected_regions.plot(figsize=(10, 8), alpha = 0.5, facecolor = 'white', edgecolor = 'black')
        #ax = self.lorraine.plot(figsize = (10, 8), alpha = 0.5, facecolor = 'white', edgecolor = 'black')


        #print(f'This is the printout of selected regions crs: {self.selected_regions.crs}')
        print(f' This is the printout of country crs: {country.crs}')
        #print(self.selected_regions.head())
        #print(self.lorraine.head())
        #print(country.head())

        #print(self.selected_regions.geom_type)
        #print("Creating Clipped...")

        #clipped = self.lorraine['geometry']   
        #p = PolygonPatch(clipped.all(),transform=ax.transData)
        #ax.set_clip_path(p)

        #ax.set_xlim(self.lorraine.total_bounds[0], self.lorraine.total_bounds[2])
        #ax.set_ylim(self.lorraine.total_bounds[1], self.lorraine.total_bounds[3])

        print("")
        print("Clipped created...")


        print("testing geom..")
        
        kde = sns.kdeplot(
            x = country.geometry.x,
            y = country.geometry.y,
            cmap = 'viridis',
            fill = True,
            alpha = 0.5,
            bw_adjust = 3,
            levels = levels
        )

        # Convert the selected regions to a MultiPolygon
        #polygon = MultiPolygon(list(self.selected_regions['geometry']))

        # Create a polygon patch from the MultiPolygon
        #p = PolygonPatch(polygon, transform=ax.transData)

        # Set the clip path to the polygon patch
        #ax.set_clip_path(p)

        #clip_path = mpath.Path(self.selected_regions.geometry.iloc[0].exterior.coords)
        #ax.set_clip_path(clip_path)       

        #p = PolygonPatch(self.selected_regions['geometry'].iloc[0],transform=ax.transData)
        #ax.set_clip_path(p)

        #ax.set_xlim(self.selected_regions.total_bounds[0], self.selected_regions.total_bounds[2])
        #ax.set_ylim(self.selected_regions.total_bounds[1], self.selected_regions.total_bounds[3])

        #contextily.add_basemap(ax, crs = 'EPSG:4326')

        plt.show()

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
        df = pd.DataFrame(level_polygons, columns =['level', 'geometry'])
        # Convert to a GeoDataFrame
        geo = gpd.GeoDataFrame(df, geometry='geometry', crs = country.crs)
        print(geo.crs)
        # Set CRS for geometric operations
        geo = geo.to_crs(epsg=3035)
        print(geo.crs)
        # Calculate area
        geo['area'] = geo['geometry'].area
        # Save to file
        geo.to_file('geo_file_TESTING.gpkg', driver='GPKG')



    

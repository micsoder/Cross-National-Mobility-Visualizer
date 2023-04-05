import pandas as pd 
import geopandas as gpd
import seaborn as sns
import contextily
import matplotlib.pyplot as plt
import folium
import pyproj


class KdeDataHandler():
    
    def __init__(self, countries):
        self.countries = countries
        self.cntr_od_tuple = self.__get_cntr_od(self.countries)
        self.cntr_id_country = self.__get_cntr_id(self.countries)


    def start_processing(self):
        
        print("Data processing starting...")
        self.df = self.__csv_to_df()        
        self.country_pair_df = self.__create_df_of_cntr_od(self.cntr_od_tuple)
        self.country_pair_gdf = self.__create_country_pair_gdf()
        self.country_1_coordinates = self.__country_coords_gdf(self.countries[0])
        self.country_2_coordinates = self.__country_coords_gdf(self.countries[1])
        self.gpkg_file = self.read_gpkg_file()
        print("Data processing done...")


    def visualize(self):
        print("Visualization starting...")
        #print(self.country_1_coordinates.head())
        #self.kde_plot(self.country_1_coordinates)
        self.region_viz(self.cntr_id_country[0], self.country_1_coordinates)
        print("Visualization done...")

    # Reading in the csv file
    def __csv_to_df(self):
        
        return pd.read_csv('mobility_data.csv', sep = ',')
        

    def __get_cntr_od(self, countries):

        country_pair1 = f"{countries[0]}_{countries[1]}"
        country_pair2 = f"{countries[1]}_{countries[0]}" #avoiding error in case of wrong order

        return country_pair1, country_pair2

    def __get_cntr_id(self, countries):
        country1 = countries[0]
        country2 = countries[1]

        return country1, country2

    def __create_df_of_cntr_od(self, cntr_od_tuple):

        return self.df.loc[self.df['CNTR_OD'].isin([cntr_od_tuple[0], cntr_od_tuple[1]])]
        
    def __create_country_pair_gdf(self):

        gdf_start = self.__df_to_gdf(self.country_pair_df.h3_grid_res10_start_lon, self.country_pair_df.h3_grid_res10_start_lat)
        gdf_end = self.__df_to_gdf(self.country_pair_df.h3_grid_res10_end_lon, self.country_pair_df.h3_grid_res10_end_lat)

        country_pair_gdf = self.__combine_gdfs(gdf_start, gdf_end)

        return country_pair_gdf

    def __df_to_gdf(self, lon, lat):

        gdf = gpd.GeoDataFrame(
        self.country_pair_df, geometry = gpd.points_from_xy(lon, lat))

        print(f'This is gdf crs before setting a new: {gdf.crs}')
        gdf = gdf.set_crs(epsg=3857)
        gdf = gdf.to_crs(3857)
        print(f'This is gdf crs after setting a new: {gdf.crs}')


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

        return country_gdf
    
    def kde_plot(self, country):

        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the kernel density estimate of start locations as a contour plot
        sns.kdeplot(
            x = country.geometry.x,
            y = country.geometry.y,
            cmap = 'viridis',
            fill = True,
            alpha = 0.5,
            ax = ax,
            levels = self.contour_intervalls(8)
        )
        

        # Specify the extent of the basemap using the total bounds of the data
        extent = country.total_bounds
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])


        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Kernel Density Plot of Mobility Data')

        #contextily.add_basemap(ax, crs = 'EPSG:4326')

        plt.show()


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


        self.border_data = gpd.read_file('GRL_region.gpkg')

        self.border_data = self.border_data.to_crs(epsg = 4326)
    
    def region_viz(self, country1, country):
        
        self.selected_regions = self.border_data.loc[self.border_data['FIPS'].isin([country1])]

        ax = self.selected_regions.plot(figsize=(10, 8), alpha = 0.5, facecolor = 'white', edgecolor = 'black')

        print(f'This is the printout of selected regions crs: {self.selected_regions.crs}')
        print(f' This is the printout of country crs: {country.crs}')

        print(self.selected_regions.head())
        print(country.head())

        sns.kdeplot(
            x = country.geometry.x,
            y = country.geometry.y,
            cmap = 'viridis',
            fill = True,
            alpha = 0.5,
            ax = ax,
            levels = self.contour_intervalls(4)
        )


        contextily.add_basemap(ax, crs = 'EPSG:4326')

        plt.show()



        #extent = self.selected_regions.total_bounds
        #regions_points = country.geometry.cx[extent[0]:extent[2], extent[1]:extent[3]]   
        
        #ax.set_xlim(extent[0], extent[2])
        #ax.set_ylim(extent[1], extent[3])

        #contextily.add_basemap(ax, crs = 'EPSG:3857')
        #ax.set_axis_off()
        









import pandas as pd 
import geopandas as gpd

class KdeDataHandler():
    
    def __init__(self, countries):
        self.countries = countries
        self.cntr_od_tuple = self.__get_cntr_od(self.countries)


    def start_analysis(self):

        self.df = self.__csv_to_df()        
        self.country_pair_df = self.__create_df_of_cntr_od(self.cntr_od_tuple)
        self.country_pair_gdf = self.__create_country_pair_gdf()


    def visualize(self):
        
        print(self.country_pair_gdf.head())
        print(self.country_pair_gdf['CNTR_OD'].head()) 

    def __csv_to_df(self):
        
        return pd.read_csv('mobility_data.csv', sep = ',')
        

    def __get_cntr_od(self, countries):

        country_pair1 = f"{countries[0]}_{countries[1]}"
        country_pair2 = f"{countries[1]}_{countries[0]}" #avoiding error in case of wrong order

        return country_pair1, country_pair2

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

        gdf = gdf.set_crs(epsg=3857)

        return gdf
        
    def __combine_gdfs(self, gdf_start, gdf_end):

        country_pair_gdf = gdf_start.set_geometry('geometry').rename(columns={'geometry':'geometry_of_start'})
        country_pair_gdf['geometry_of_end'] = gdf_end['geometry']

        return country_pair_gdf



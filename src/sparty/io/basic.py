import pandas as pd
import spatialdata as sd
import spatialdata_io
import math 

def load_merscope(
    path: str,
    slide_name: str,
    vpt_outputs: str = None,
    region_name: str = "region_0",
    z_layers: int = 2,
    feature_key: str = "gene",
    # instance_key: str = 'cell_id',
    table_key: str = 'table',
    # spatial_key: str = 'spatial',
    layer: str ='counts',
) -> sd.SpatialData:
    """Load vizgen merscope data as SpatialData object

    Parameters
    ----------
    path
        path to folder.
    vpt_outputs
        path to vpt folder.
    region_name
        region_name id.
    slide_name
        slide_name id.
    z_layers
        z layers to load.
    feature_key
        default column for feature name in transcripts.
        
    Returns
    -------
    SpatialData object
    """
    sdata = spatialdata_io.merscope(
        path=path, vpt_outputs=vpt_outputs, region_name=region_name, 
        slide_name=slide_name, z_layers=z_layers
    )

    sdata[table_key].obs_names.name = None
    sdata[table_key].layers[layer] = sdata[table_key].X.copy()
    # sdata[table_key].obs[["center_x", "center_y"]] = sdata[table_key].obsm[spatial_key]

    sdata[table_key].uns["spatialdata_attrs"]["feature_key"] = feature_key
    sdata[table_key].uns["technology"] = "merscope"

    # sdata['table'].uns["spatialdata_attrs"]["region"] = region
    # sdata['table'].obs["region"] = region
    # sdata['table'].obs["region"] = sdata['table'].obs["region"].astype('category')

    # if not sdata.locate_element(sdata.table.uns["spatialdata_attrs"]["region"]) == []:
    #    sdata[sdata.table.uns["spatialdata_attrs"]["region"]].index.name = None
    
    # Transform coordinates to mosaic pixel coordinates
    # transformation_matrix = pd.read_csv(
    #    path + "/images/micron_to_mosaic_pixel_transform.csv", header=None, sep=" "
    # ).values
    # temp = sdata.table.obs[["center_x", "center_y"]].values
    # cell_positions = np.ones((temp.shape[0], temp.shape[1] + 1))
    # cell_positions[:, :-1] = temp
    # transformed_positions = np.matmul(transformation_matrix, np.transpose(cell_positions))[:-1]
    # sdata.table.obs["center_x_pix"] = transformed_positions[0, :]
    # sdata.table.obs["center_y_pix"] = transformed_positions[1, :]
    # sdata.table.obs = sdata.table.obs.drop(columns=["min_x", "max_x", "min_y", "max_y"])

    # coord_pixels = sdata.table[["center_x_pix", "center_x_pix"]].to_numpy()
    # coord_microns = sdata.table[["center_x", "center_y"]].to_numpy()
    # sdata.table.obsm={"microns": coord_microns, "pixels": coord_pixels},

    # percent_in_cell = sdata.table.obs.n_Counts.sum(axis=0) * 100 / len(sdata[key + '_transcripts'])
    # print("\n" + slide_name)
    # print("total cells=", sdata.table.obs.shape[0])
    # print("total transcripts=", len(sdata[key + '_transcripts']))
    # print("% in cells=", percent_in_cell)
    # print("mean transcripts per cell=", sdata.table.obs["n_Counts"].mean())
    # print("median transcripts per cell=", sdata.table.obs["n_Counts"].median())

    return sdata


def load_xenium(
    path: str,
    layer: str ='counts',
    region: str = "cell_boundaries",
    feature_key: str = "feature_name",
    instance_key: str = 'cell_id',
    table_key: str = 'table',
    spatial_key: str = 'spatial',
    index_table: bool = True,
    n_jobs: int = 1,
) -> sd.SpatialData:
    """
    Load xenium data as SpatialData object

    Parameters
    ----------
    path
        path to folder.
    index_table
        rename the index in table.obs with the cell_id
    region
        default shape element for region in table.obs.
    feature_key
        default column for feature name in transcripts.
    n_jobs
        number of jobs to load the xenium object

    Returns
    -------
    SpatialData object
    """
    sdata = spatialdata_io.xenium(path, n_jobs = n_jobs)
    sdata[table_key].layers[layer] = sdata[table_key].X.copy()
    sdata[table_key].obs[["center_x", "center_y"]] = sdata[table_key].obsm[spatial_key]
    
    sdata[table_key].uns["spatialdata_attrs"] = {
        'region': region,
        'region_key': 'region',
        'instance_key': instance_key,
        'feature_key': feature_key 
    }
    sdata[table_key].uns["technology"] = "xenium"

    sdata[table_key].obs['region'] = region
    sdata[table_key].obs['region'] = sdata[table_key].obs['region'].astype('category')

    if index_table:
        sdata[table_key].obs_names = sdata[table_key].obs[instance_key].values

    return sdata


def load_cosmx(
    path: str,
    dataset_id: str = "R5941_ColonTMA",
    feature_key: str = "target",
    table_key: str = 'table',
    spatial_key: str = 'spatial',
) -> sd.SpatialData:
    """
    Load cosmx data as SpatialData object

    Parameters
    ----------
    path
        path to folder.
    dataset_id
        dataset_id.
    feature_key
        default column for feature name in transcripts.

    Returns
    -------
    SpatialData object
    """
    sdata = spatialdata_io.cosmx(path, dataset_id=dataset_id, transcripts=True)
    sdata[table_key].layers["counts"] = sdata[table_key].X.copy()
    sdata[table_key].obs[["center_x", "center_y"]] = sdata[table_key].obsm[spatial_key]

    sdata[table_key].uns["spatialdata_attrs"]["feature_key"] = feature_key
    sdata[table_key].uns["technology"] = "cosmx"
    # sdata.table.uns["spatialdata_attrs"]["region"] = region

    return sdata


# make a class load_data
# parameter -> techno = xenium merscope or cosmx
# xenium = {"function" : spatialdata_io.xenium(path, n_jobs = n_jobs),
#           "region" : region,
#           "feature_key" : "feature_name"}


def subsetSample(
    sdata: sd.SpatialData,
    sample: str,
    regions: list,
    manip: dict,
    fov_locations: pd.DataFrame,
    scale_factor: float = 0.2125,
    region: str = 'cell_boundaries',
) -> sd.SpatialData:
    """
    Separe samples based on fov locations

    Parameters
    ----------
        to be completed

    Returns
    -------
        to be completed
    """
    manip['Region name'] = sample
    width = fov_locations.iloc[0]['width']
    height = fov_locations.iloc[0]['height']

    print(f'Start with: {sample}')
    x_min = math.ceil(fov_locations.loc[regions,]['x'].min() / scale_factor)
    x_max = math.ceil((fov_locations.loc[regions,]['x'].max() + width) / scale_factor)   
    y_min = math.ceil(fov_locations.loc[regions,]['y'].min() / scale_factor)
    y_max = math.ceil((fov_locations.loc[regions,]['y'].max() + height) / scale_factor) 
    print(f'xmin: {x_min}, xmax: {x_max}')
    print(f'ymin: {y_min}, ymax: {y_max}')
        
    print("Start subset region...")
    sub_sdata = sd.bounding_box_query(sdata,
                                      min_coordinate=[x_min, y_min],
                                      max_coordinate=[x_max, y_max],
                                      axes=("x", "y"),
                                      target_coordinate_system="global")
    sub_sdata['table'].uns["spatialdata_attrs"]["region"] = region
    sub_sdata['table'].obs['region'] = region
    print(sub_sdata['table'].uns["spatialdata_attrs"])
    print("Done subset region")
    return sub_sdata

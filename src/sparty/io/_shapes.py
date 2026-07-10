import pandas as pd
import geopandas as gpd
import shapely


from ..pl._shapes import plot_shapes

def shapes_from_xe(
    file,
    pixel_size: float = 0.2125,
    plot_fig: bool = True,
    ncols: int = 4,
    return_gdf: bool = True, 
):
    """
    Create polygons from a CSV file containing X/Y coordinates.

    If the CSV has a 'Selection' column, one polygon is created per selection.
    Otherwise, the whole file is treated as one polygon.
    """
    df = pd.read_csv(file, skiprows=2)

    if not {"X", "Y"}.issubset(df.columns):
        raise ValueError("CSV file must contain 'X' and 'Y' columns.")

    if pixel_size <= 0:
        raise ValueError("pixel_size must be greater than 0.")

    if "Selection" not in df.columns:
        df["Selection"] = "shape"

    shapes = {}

    for name, group in df.groupby("Selection", sort=False):
        coords = group[["X", "Y"]] / pixel_size
        shapes[name] = shapely.Polygon(coords.to_numpy())
    
    if plot_fig:
        plot_shapes(shapes, ncols=ncols)
       
    if not return_gdf:
        return shapes
    
    return gpd.GeoDataFrame({
        "name": shapes.keys(),
        "geometry": shapes.values(),},
        geometry="geometry",
    )
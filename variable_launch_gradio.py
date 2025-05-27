import argparse
from launch_gradio import *


def process_so2_data(paths):
    latitudes = ["30S(Tg)", "15S(Tg)", "15N(Tg)", "30N(Tg)"]
    data = [pd.read_csv(path, sep="\s+").set_index("Timestamp")[latitudes].loc["2035":"2070"] for path in paths]
    # Slice each between 2035 and 2070
    data = [df.loc["2035":"2070"] for df in data]
    # Take mean of the three members
    data = sum(data) / len(data)
    # Slice to latitude values
    data = data[latitudes]
    # data is a pandas array with columns as latitudes and rows as time
    # Convert to an xarray dataset with dimensions lat and time
    data = xr.Dataset(data)
    data = data.rename({"Timestamp": "time"})
    # latitudes are data variables, so convert to coordinates
    data = data.to_array(dim="lat")
    return data


def generate_gradio_plots(ssp_scenario, spatial_agg, spatial_item, decade_visualization, *variable_injection_amounts):
    """
    Create plots for gradio.
    """
    decade_start_year, decade_end_year = map(int, decade_visualization.split("-"))

    if spatial_agg != "CESM2-WACCM 1 Degree Lat/Lon Grid":
        spatial_gdf = SPATIALAGG2GDF[spatial_agg]

    else:
        spatial_gdf = None

    variable_injection_amounts = np.array(variable_injection_amounts).reshape(7, 7)
    # Repeat per decade to create a (7, 6) array
    first_decade = variable_injection_amounts[:, 0][:, np.newaxis]
    # Repeat the first decade 6 times
    first_decade = np.repeat(first_decade, 6, axis=1)
    other_decades = variable_injection_amounts[:, 1:]
    # Repeat the other decades 10 times
    other_decades = np.repeat(other_decades, 10, axis=1)
    variable_injection_amounts = np.concatenate([first_decade, other_decades], axis=1)
    vars = ["tas", "p-e"]
    output_data = get_outputs(ssp_scenario, None, spatial_gdf, spatial_item,
                              decade_start_year, decade_end_year, None, None,
                              DATA_DIR, MODEL_DIR, CACHE_DIR, var=vars,
                              variable_injection=variable_injection_amounts)

    if USE_ROBINSON_PROJECTION:
        projection = ccrs.Robinson()
    else:
        projection = ccrs.PlateCarree()

    var2plots = {}

    for var in vars:

        var_plots = {}
        title, simple_title, cmap, boundary_color = VAR2INFO[var]
        if var in ["tasmin", "tasmax"]:
            title += " (Delta from PI)"

        vmin, vmax = get_vmin_vmax(var, var in ["tas", "tasmin", "tasmax"], False)
        delta_vmin, delta_vmax = get_vmin_vmax(var, var in ["tas", "tasmin", "tasmax"], True)

        #### Regional map plots ####
        regional_fig, regional_axs = plt.subplots(1, 1, figsize=REGIONAL_FIGSIZE, subplot_kw={'projection': projection})    
        sai_fig, sai_axs = plt.subplots(1, 1, figsize=REGIONAL_FIGSIZE, subplot_kw={'projection': projection})
        delta_fig, delta_axs = plt.subplots(1, 1, figsize=REGIONAL_FIGSIZE, subplot_kw={'projection': projection})

        var_plots["delta"] = delta_fig

        regional_mean = output_data[var]["regional_mean"]
        regional_delta_mean = output_data[var]["regional_delta_mean"]

        if spatial_gdf is not None and spatial_item is not None:
            mask = output_data[var]["mask"]

            regional_mean.where(~mask).plot(ax=regional_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=False, alpha=0.3, transform=ccrs.PlateCarree())
            regional_mean.where(mask).plot(ax=regional_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=True, alpha=1, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})

            regional_delta_mean.where(~mask).plot(ax=delta_axs, cmap=cmap if var != "pr" else "BrBG", robust=True, vmin=delta_vmin, vmax=delta_vmax, add_colorbar=False, alpha=0.3, transform=ccrs.PlateCarree())
            regional_delta_mean.where(mask).plot(ax=delta_axs, cmap=cmap if var != "pr" else "BrBG", robust=True, vmin=delta_vmin, vmax=delta_vmax, add_colorbar=True, alpha=1, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})

            (regional_mean + regional_delta_mean).where(~mask).plot(ax=sai_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=False, alpha=0.3, transform=ccrs.PlateCarree())
            (regional_mean + regional_delta_mean).where(mask).plot(ax=sai_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=True, alpha=1, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})

        else:
            regional_mean.plot(ax=regional_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=True, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})
            regional_delta_mean.plot(ax=delta_axs, cmap=cmap if var != "pr" else "BrBG", robust=True, vmin=delta_vmin, vmax=delta_vmax, add_colorbar=True, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})
            (regional_mean + regional_delta_mean).plot(ax=sai_axs, cmap=cmap, robust=True, vmin=vmin, vmax=vmax, add_colorbar=True, transform=ccrs.PlateCarree(), cbar_kwargs={'extend': 'neither'})

        # Add hatches where p values are above 0.05
        regional_no_sai_p_values = output_data[var]["regional_no_sai_p_values"]
        regional_sai_p_values = output_data[var]["regional_sai_p_values"]
        if regional_no_sai_p_values is not None and regional_sai_p_values is not None:
            if spatial_gdf is not None and spatial_item is not None:
                mask = output_data[var]["mask"]

                hatches = np.where((~mask) & (regional_no_sai_p_values >= 0.05), 1, np.nan)
                masked_hatches = np.ma.masked_where(np.isnan(hatches), hatches)
                regional_axs.contourf(
                    regional_mean['lon'], regional_mean['lat'], masked_hatches,
                    levels=[0.5, 1.5],  # Contour levels (must span the values in `masked_hatches`)
                    hatches=['//'],  # Apply hatches only to valid regions
                    colors='none',  # Prevent filling with solid colors
                    transform=ccrs.PlateCarree()
                )

                hatches = np.where((~mask) & (regional_sai_p_values >= 0.05), 1, np.nan)
                masked_hatches = np.ma.masked_where(np.isnan(hatches), hatches)
                sai_axs.contourf(
                    regional_mean['lon'], regional_mean['lat'], masked_hatches,
                    levels=[0.5, 1.5],  # Contour levels (must span the values in `masked_hatches`)
                    hatches=['//'],  # Apply hatches only to valid regions
                    colors='none',  # Prevent filling with solid colors
                    transform=ccrs.PlateCarree()
                )
            else:
                hatches = np.where((~np.isnan(regional_no_sai_p_values)) & (regional_no_sai_p_values >= 0.05), 1, np.nan)
                masked_hatches = np.ma.masked_where(np.isnan(hatches), hatches)
                regional_axs.contourf(
                    regional_no_sai_p_values['lon'], regional_no_sai_p_values['lat'], masked_hatches,
                    levels=[0.5, 1.5],  # Contour levels (must span the values in `masked_hatches`)
                    hatches=['//'],  # Apply hatches only to valid regions
                    colors='none',  # Prevent filling with solid colors
                    transform=ccrs.PlateCarree()
                )

                hatches = np.where((~np.isnan(regional_sai_p_values)) & (regional_sai_p_values >= 0.05), 1, np.nan)
                masked_hatches = np.ma.masked_where(np.isnan(hatches), hatches)
                sai_axs.contourf(
                    regional_sai_p_values['lon'], regional_sai_p_values['lat'], masked_hatches,
                    levels=[0.5, 1.5],  # Contour levels (must span the values in `masked_hatches`)
                    hatches=['//'],  # Apply hatches only to valid regions
                    colors='none',  # Prevent filling with solid colors
                    transform=ccrs.PlateCarree()
                )

        regional_axs_title = f'No SAI {simple_title} Delta from PI ({ssp_scenario}) from {decade_start_year} to {decade_end_year}'
        delta_axs_title = f'SAI {simple_title} Delta from PI from {decade_start_year} to {decade_end_year}'
        sai_axs_title = f'SAI {simple_title} Delta from PI ({ssp_scenario}) from {decade_start_year} to {decade_end_year}'
        if var not in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
            regional_axs_title = regional_axs_title.replace(" Delta from PI", "")
            delta_axs_title = delta_axs_title.replace(" Delta from PI", "")
            sai_axs_title = sai_axs_title.replace(" Delta from PI", "")
        regional_axs.set_title(regional_axs_title, fontsize=FONTSIZE)
        delta_axs.set_title(delta_axs_title, fontsize=FONTSIZE)
        sai_axs.set_title(sai_axs_title, fontsize=FONTSIZE)

        # Always overlay continental boundaries in a faded way
        SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=regional_axs, linewidth=0.5, edgecolor='black', alpha=0.3, transform=ccrs.PlateCarree())
        SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=delta_axs, linewidth=0.5, edgecolor='black', alpha=0.3, transform=ccrs.PlateCarree())
        SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=sai_axs, linewidth=0.5, edgecolor='black', alpha=0.3, transform=ccrs.PlateCarree())

        if spatial_gdf is None:
            SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=regional_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
            SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=delta_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
            SPATIALAGG2GDF["IPCC-WGII-continental-regions"].boundary.plot(ax=sai_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
        else:
            # Plot all the boundaries
            spatial_gdf.boundary.plot(ax=regional_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
            spatial_gdf.boundary.plot(ax=delta_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
            spatial_gdf.boundary.plot(ax=sai_axs, edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree())
            if spatial_item is not None:
                # Plot the selected region in red
                spatial_gdf[spatial_gdf.name == spatial_item].boundary.plot(ax=regional_axs, edgecolor=boundary_color, linewidth=0.5, transform=ccrs.PlateCarree())
                spatial_gdf[spatial_gdf.name == spatial_item].boundary.plot(ax=delta_axs, edgecolor=boundary_color, linewidth=0.5, transform=ccrs.PlateCarree())
                spatial_gdf[spatial_gdf.name == spatial_item].boundary.plot(ax=sai_axs, edgecolor=boundary_color, linewidth=0.5, transform=ccrs.PlateCarree())

        regional_fig.tight_layout()
        sai_fig.tight_layout()
        delta_fig.tight_layout()

        var_plots["slider"] = (fig_to_image(regional_fig), fig_to_image(sai_fig))

        if var not in ['pr', 'p-e'] or spatial_item is not None:
            global_mean_fig, global_mean_ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)
        else:
            global_mean_fig = None
            global_mean_ax = None

        pdf_fig, pdf_ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)

        #### Mean over time plot ####
        mean_no_sai = output_data[var]["mean_over_time"]["no_sai"]
        mean_with_sai = output_data[var]["mean_over_time"]["with_sai"]
        global_injection_amounts = variable_injection_amounts.sum(axis=0)
        if np.any(global_injection_amounts):
            # If there is at least one nonzero (any SAI is done)
            # Slice to the year of the first nonzero injection
            first_nonzero_index = np.argmax(global_injection_amounts > 0)
            first_nonzero_year = MIN_SAI_START + first_nonzero_index
            mean_with_sai = mean_with_sai.sel(time=slice(first_nonzero_year, 2101))

        historical_model_global_mean = output_data[var]["mean_over_time"]["historical_model"]

        if global_mean_ax is not None:
            global_mean_ax.plot(mean_no_sai.time.values, mean_no_sai.values, label='No SAI', color='tab:red', linestyle='--')
            global_mean_ax.plot(mean_with_sai.time.values, mean_with_sai.values, label='With SAI', color='tab:blue', linestyle='--')
            global_mean_ax.plot(historical_model_global_mean.time.values, historical_model_global_mean.values, label='Historical CESM2-WACCM', color='tab:red')

            if "natural_variability" in output_data[var]["mean_over_time"]:
                # Add shading of natural variability around each of the above 3 lines
                natural_variability = output_data[var]["mean_over_time"]["natural_variability"]
                global_mean_ax.fill_between(mean_no_sai.time.values, mean_no_sai.values - natural_variability*2, mean_no_sai.values + natural_variability*2, color='tab:red', alpha=0.3)
                global_mean_ax.fill_between(mean_with_sai.time.values, mean_with_sai.values - natural_variability*2, mean_with_sai.values + natural_variability*2, color='tab:blue', alpha=0.3)
                global_mean_ax.fill_between(historical_model_global_mean.time.values, historical_model_global_mean.values - natural_variability*2, historical_model_global_mean.values + natural_variability*2, color='tab:red', alpha=0.3)

            if var == 'tas':
                historical_obs_data = output_data[var]["mean_over_time"]["historical_obs"]
                global_mean_ax.plot(historical_obs_data.time.values, historical_obs_data.values, label='Historical Observations', color='black')
            
            global_mean_ax.set_xlabel('Year', fontsize=FONTSIZE)
            global_mean_ax.set_ylabel(title.replace("Delta", "Delta from Preindustrial"), fontsize=FONTSIZE)
            global_mean_ax.set_xticks(np.arange(1850, 2101, 25))
            # Add vertical lines for selected time range
            sai_start_year = mean_with_sai.time[0].item()
            global_mean_ax.axvline(sai_start_year, color='black', linestyle='--')
            global_mean_ax.axvline(SIM_END_YEAR, color='black', linestyle='-')
            # Add "SAI Start" and "Simulation End" labels directly on top of the vertical lines
            global_mean_ax.text(sai_start_year, global_mean_ax.get_ylim()[1], "SAI Start", fontsize=FONTSIZE, ha='center')
            global_mean_ax.text(SIM_END_YEAR, global_mean_ax.get_ylim()[1], "Simulation End", fontsize=FONTSIZE, ha='center')

            global_mean_ax.legend()

        if global_mean_fig is not None:
            global_mean_fig = gr.Plot(global_mean_fig, visible=True)
        else:
            global_mean_fig = gr.Plot(visible=False)

        var_plots["global_mean"] = global_mean_fig

        #### PDF plot ####
        no_sai_counts = output_data[var]["distribution"]["no_sai"]["counts"]
        no_sai_bins = output_data[var]["distribution"]["no_sai"]["bin_edges"]
        with_sai_counts = output_data[var]["distribution"]["with_sai"]["counts"]
        with_sai_bins = output_data[var]["distribution"]["with_sai"]["bin_edges"]
        if "historical" in output_data[var]["distribution"]:
            historical_counts = output_data[var]["distribution"]["historical"]["counts"]
            historical_bins = output_data[var]["distribution"]["historical"]["bin_edges"]

        if "above" in var or "below" in var:
            bar_width = 0.25  # width of each bar
            # Convert bins to string ranges, inclusive of the lft bin and exclusive of the right bin
            no_sai_bins = [f"{int(no_sai_bins[i])}-{int(no_sai_bins[i+1])-1}" for i in range(len(no_sai_bins)-2)] + [f"{int(no_sai_bins[-2])}+"]
            with_sai_bins = [f"{int(with_sai_bins[i])}-{int(with_sai_bins[i+1])-1}" for i in range(len(with_sai_bins)-2)] + [f"{int(with_sai_bins[-2])}+"]
            index = np.arange(len(no_sai_bins))
            pdf_ax.bar(index, height=no_sai_counts, width=bar_width, alpha=0.5, label=f'No SAI  ({decade_start_year}-{decade_end_year})', color='tab:red')
            pdf_ax.bar(index+bar_width, height=with_sai_counts, width=bar_width, alpha=0.5, label=f'With SAI  ({decade_start_year}-{decade_end_year})', color='tab:blue')
            if "historical" in output_data[var]["distribution"]:
                historical_bins = [f"{int(historical_bins[i])}-{int(historical_bins[i+1])-1}" for i in range(len(historical_bins)-2)] + [f"{int(historical_bins[-2])}+"]
                pdf_ax.bar(index+2*bar_width, height=historical_counts, width=bar_width, alpha=0.5, label='Historical Observations', color='black')
            pdf_ax.set_xticks(index + bar_width)
            pdf_ax.set_xticklabels(no_sai_bins)
            # ylabel = 'Log Frequency'
            ylabel = 'Frequency'
        else:
            no_sai_bins = no_sai_bins[:-1]
            with_sai_bins = with_sai_bins[:-1]
            pdf_ax.hist(no_sai_bins, bins=no_sai_bins, weights=no_sai_counts, alpha=0.5, label=f'No SAI  ({decade_start_year}-{decade_end_year})', color='tab:red', density=False)
            pdf_ax.hist(with_sai_bins, bins=with_sai_bins, weights=with_sai_counts, alpha=0.5, label=f'With SAI  ({decade_start_year}-{decade_end_year})', color='tab:blue', density=False)
            if "historical" in output_data[var]["distribution"]:
                historical_bins = historical_bins[:-1]
                pdf_ax.hist(historical_bins, bins=historical_bins, weights=historical_counts, alpha=0.5, label='Historical Observations', color='black', density=False)
            ylabel = 'Frequency'

        pdf_ax.set_xlabel(title.replace("Delta", "Delta from Preindustrial"), fontsize=FONTSIZE)
        pdf_ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        pdf_ax.legend()

        var_plots["pdf"] = pdf_fig

        var2plots[var] = var_plots

    ### Latitude vs. Tg SO2 plot ###
    so2_by_latitude = output_data["so2_by_latitude"]
    so2_by_latitude_fig, so2_by_latitude_ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)
    # Custom color map from beige to blue
    colors = ["#f5f5dc", "#000080"]  # Beige to Navy Blue
    cmap = LinearSegmentedColormap.from_list("CustomMap", colors, N=256)
    so2_by_latitude_df = so2_by_latitude.to_dataframe("so2").unstack(level='time')
    so2_by_latitude_df = so2_by_latitude_df.droplevel(level=0, axis=1)
    so2_by_latitude_df = so2_by_latitude_df.iloc[::-1]
    sns.heatmap(so2_by_latitude_df, annot=False, cmap=cmap, cbar_kws={'label': 'Tg SO2'}, ax=so2_by_latitude_ax)
    so2_by_latitude_ax.set_title('Latitude vs. Tg SO2')
    so2_by_latitude_ax.set_xlabel('Year')
    so2_by_latitude_ax.set_ylabel('Latitude')
    so2_by_latitude_fig.tight_layout()

    ### Global SO2 plot ###
    global_so2 = output_data["global_so2"]
    global_so2_fig, global_so2_ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)
    global_so2.plot(ax=global_so2_ax)
    total_SO2 = global_so2.sum()
    global_so2_ax.set_title(f'Global SO2 (Total: {total_SO2:.2f} Tg SO2)')
    global_so2_ax.set_xlabel('Year')
    global_so2_ax.set_ylabel('Tg SO2')
    global_so2_fig.tight_layout()

    # Return all the plots
    # by using a for loop over the variables
    outputs = []
    for var in var2plots:
        outputs.append(ImageSlider(var2plots[var]["slider"]))
        outputs.append(var2plots[var]["delta"])
        outputs.append(var2plots[var]["global_mean"])
        outputs.append(var2plots[var]["pdf"])

    outputs.append(so2_by_latitude_fig)
    outputs.append(global_so2_fig)

    return tuple(outputs)


if __name__ == "__main__":

    params = get_params()
    CACHE_DIR = params["CACHE_DIR"]
    DATA_DIR = params["DATA_DIR"]
    MODEL_DIR = params["MODEL_DIR"]
    USE_ROBINSON_PROJECTION = params["USE_ROBINSON_PROJECTION"]
    PORT = params["PORT"]
    SHARE = params["SHARE"]
    FONTSIZE = params["FONTSIZE"]
    REGIONAL_FIGSIZE = params["REGIONAL_FIGSIZE"]
    GLOBAL_FIGSIZE = params["GLOBAL_FIGSIZE"]
    SLIDER_SIZE = params["SLIDER_SIZE"]
    SPATIALAGG2GDF = params["SPATIALAGG2GDF"]
    SPATIALAGG_ITEMS = params["SPATIALAGG_ITEMS"]
    DECADES = params["DECADES"]

    with gr.Blocks(title="SAI Demo V2.0") as demo:
        with gr.Row():
            dropdown_ssp = gr.Dropdown(value="SSP2-4.5", choices=FANCY_SSP_TITLES.values(), label="Select an SSP Scenario")
            dropdown_spatial_group = gr.Dropdown(value="CESM2-WACCM 1 Degree Lat/Lon Grid", choices=sorted(list(SPATIALAGG_ITEMS.keys())), label="Select spatial aggregation group")
            dropdown_spatial_item = gr.Dropdown(choices=[], label="Select specific spatial region", allow_custom_value=True)
            decade_visualization = gr.Dropdown(value="2091-2100", choices=DECADES, label="Select decade to visualize")

        dropdown_spatial_group.change(fn=update_items_dropdown, inputs=dropdown_spatial_group, outputs=dropdown_spatial_item)

        variable_inj_labels = ["60N", "30N", "15N", "0NS", "15S", "30S", "60S"]
        decades = ["2035-2040"] + [f"{decade}-{decade+9}" for decade in range(2041, 2092, 10)]
        variable_inj_sliders = []
        for variable_inj_label in variable_inj_labels:
            with gr.Row():
                gr.Markdown(f"# {variable_inj_label}")
                for decade in decades:
                        variable_inj_sliders.append(gr.Slider(minimum=0, maximum=10, value=0, step=1, label=decade))

        btn = gr.Button("Submit")
        outputs = [
            ImageSlider(label="Regional Temperature", type="pil", slider_color="blue"),
            gr.Plot(label="Regional Temperature | No SAI vs. SAI"),
            gr.Plot(label="Mean Temperature Over Time"),
            gr.Plot(label="Temperature Distribution")
        ]
        other_variables = [
            # ("Precipitation", "red"),
            ("Water Availability", "red"),
        ]
        for var, slider_color in other_variables:
            outputs.append(ImageSlider(label=f"Regional {var}", type="pil", slider_color=slider_color))
            outputs.append(gr.Plot(label=f"Regional {var} | No SAI vs. SAI"))
            outputs.append(gr.Plot(label=f"Mean {var} Over Time", visible=var != "Precipitation"))
            outputs.append(gr.Plot(label=f"{var} Distribution"))

        outputs.append(gr.Plot(label="Latitude vs. Tg SO2"))
        outputs.append(gr.Plot(label="Global SO2"))
        btn.click(
            fn=generate_gradio_plots,
            inputs=[dropdown_ssp, dropdown_spatial_group, dropdown_spatial_item, decade_visualization] + variable_inj_sliders,
            outputs=outputs
        )

    demo.launch(share=SHARE, server_name="0.0.0.0", server_port=PORT)

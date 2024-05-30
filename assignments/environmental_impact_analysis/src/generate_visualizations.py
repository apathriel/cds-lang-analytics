from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

from data_processing import load_csv_as_df
from logger_utils import get_logger


logger = get_logger(__name__)


def visualize_emissions_by_project(
    df: pd.DataFrame,
    output_dir: Path,
    project_column: str = "project_name",
    emissions_column: str = "emissions",
    save_visualization: bool = True,
    visualization_type: str = "bar",
    bar_color: str = "#3498db",
) -> None:
    """
    Generate a visualization of emissions by project.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the emissions data.
        output_dir (Path): The directory where the visualization will be saved.
        project_column (str, optional): The column name in the DataFrame that represents the project names. Defaults to "Project_name".
        emissions_column (str, optional): The column name in the DataFrame that represents the emissions values. Defaults to "Emissions".
        save_visualization (bool, optional): Whether to save the visualization as an image. Defaults to True.
        visualization_type (str, optional): The type of visualization to generate. Defaults to "bar".
        bar_color (str, optional): The color of the bars if a bar plot is generated. Defaults to "#3498db".

    Raises:
        ValueError: If an unsupported visualization type is provided.

    """
    logger.info(
        f"Generating {visualization_type} visualization of emissions by project..."
    )
    # Group the data by project and sum the emissions
    project_emissions = df.groupby(project_column)[emissions_column].sum().reset_index()

    if visualization_type == "bar":
        # Create a bar plot of the emissions by project using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=project_column,
            y=emissions_column,
            data=project_emissions,
            color=bar_color,
        )
        plt.title("Total emissions by Project")
    elif visualization_type == "pie":
        # Create a pie chart of the emissions by project
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=project_emissions[project_column],
                    values=project_emissions[emissions_column],
                    textinfo="label+percent",
                    insidetextorientation="radial",
                    domain=dict(
                        y=[0, 0.85]
                    ),  # Adjust the area of the plot in which the pie chart is drawn
                )
            ]
        )
        fig.update_layout(
            title_text="Total emissions by Project",
            showlegend=False,
            title_x=0.5,  # Center the title
            title_y=0.9,  # Position the title towards the top
        )
    else:
        raise ValueError(f"Unsupported visualization type: {visualization_type}")

    if save_visualization:
        # Save the plot as an image
        plot_path = output_dir / f"emissions_by_project_{visualization_type}.png"
        if visualization_type == "bar":
            plt.savefig(plot_path)
        else:
            fig.write_image(str(plot_path))
        logger.info(f"Visualization saved to {plot_path}")
    else:
        if visualization_type == "bar":
            plt.show()
        else:
            fig.show()


def visualize_emissions_from_subtasks_by_project(
    df: pd.DataFrame,
    output_dir: Path,
    visualization_type: str = "bar",
    save_visualization: bool = True,
) -> None:
    """
    Visualizes the emissions from subtasks by project.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        output_dir (Path): The directory where the visualizations will be saved.
        visualization_type (str, optional): The type of visualization to generate. Defaults to "bar".
        save_visualization (bool, optional): Whether to save the visualization or display it. Defaults to True.

    Raises:
        ValueError: If an unsupported visualization type is provided.
    """
    projects = df["project_name"].unique()
    for project in projects:
        project_df = df[df["project_name"] == project]
        task_emissions = project_df.groupby("Task")["emissions"].sum().reset_index()

        if visualization_type == "bar":
            plt.figure(figsize=(12, 6))
            sns.barplot(y="Task", x="emissions", data=task_emissions)
            plt.title(f"Emissions Distribution for Project: {project}")

            if save_visualization:
                plt.tight_layout()
                plot_path = output_dir / f"{project}_emissions_distribution_bar.png"
                plt.savefig(plot_path)
                print(f"Visualization saved to {plot_path}")
            else:
                plt.show()
        elif visualization_type == "pie":
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=task_emissions["Task"],
                        values=task_emissions["emissions"],
                        textinfo="label+percent",
                        insidetextorientation="radial",
                        domain=dict(
                            y=[0, 0.85]
                        ),  # Adjust the area of the plot in which the pie chart is drawn
                    )
                ]
            )
            fig.update_layout(
                title_text=f"Emission Distribution for Project: {project}",
                showlegend=False,
                title_x=0.5,  # Center the title
                title_y=0.05,  # Position the title towards the top
            )

            if save_visualization:
                plot_path = output_dir / f"{project}_emissions_distribution_pie.png"
                fig.write_image(str(plot_path))
                print(f"Visualization saved to {plot_path}")
            else:
                fig.show()
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")


def main():
    input_data_directory = Path(__file__).resolve().parents[1] / "in"
    input_data_file = "combined_emissions.csv"
    input_data_path = input_data_directory / input_data_file

    output_data_directory = Path(__file__).resolve().parents[1] / "out"
    output_data_directory.mkdir(parents=True, exist_ok=True)

    emission_data = load_csv_as_df(input_data_path, logging_enabled=False)

    visualize_emissions_by_project(
        df=emission_data,
        output_dir=output_data_directory,
        visualization_type="bar",
        save_visualization=True,
    )

    visualize_emissions_by_project(
        df=emission_data,
        output_dir=output_data_directory,
        visualization_type="pie",
        save_visualization=True,
    )

    visualize_emissions_from_subtasks_by_project(
        df=emission_data,
        output_dir=output_data_directory,
        visualization_type="bar",
        save_visualization=True,
    )

    visualize_emissions_from_subtasks_by_project(
        df=emission_data,
        output_dir=output_data_directory,
        visualization_type="pie",
        save_visualization=True,
    )

if __name__ == "__main__":
    main()

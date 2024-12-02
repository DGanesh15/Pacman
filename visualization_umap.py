import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import umap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GAME_TITLE = 'Pacman'
GAME_ID = 'MsPacmanNoFrameskip-v4'
RANDOM_SEED = 1732481853


class PacmanVisualizer:
    def __init__(self):
        # Configuration
        self.embedding_neighbors = 15
        self.embedding_components = 2

        # Create figure
        self.figure = plt.figure('Pacman UMAP View', figsize=(12, 8))
        self.figure.patch.set_facecolor("#f5f5f5")

        # Event listeners
        self.figure.canvas.mpl_connect('draw_event', self.on_canvas_draw)

        # Main UMAP plot
        self.umap_axis = plt.subplot2grid((30, 40), (0, 0), rowspan=30, colspan=30)
        self.umap_axis.set_facecolor("#1e1e1e")
        self.umap_axis.set_title("UMAP Visualization of Pacman States", fontsize=16)

        # Placeholder scatter plot
        self.umap_scatter = self.umap_axis.scatter([0], [0])

        # State image display area
        self.selected_index = 0
        self.state_image_axis = plt.subplot2grid((30, 40), (15, 33), rowspan=15, colspan=7)
        self.state_image = self.state_image_axis.imshow([[[0, 0, 0]]], interpolation='none')
        self.state_image_axis.set_xticklabels([])
        self.state_image_axis.set_yticklabels([])

        # Highlight marker
        self.highlight_marker, = self.umap_axis.plot(
            [0],
            [0],
            animated=True,
            linestyle="",
            marker="o",
            markersize=10,
            markerfacecolor="cyan",
            markeredgecolor="black",
        )

        # Buttons for navigation
        self.button_prev_ax = plt.axes([0.1, 0.01, 0.1, 0.05])
        self.button_next_ax = plt.axes([0.21, 0.01, 0.1, 0.05])
        self.button_jump_prev_ax = plt.axes([0.75, 0.83, 0.09, 0.02])
        self.button_jump_next_ax = plt.axes([0.85, 0.83, 0.09, 0.02])

        self.button_prev = Button(self.button_prev_ax, 'Previous')
        self.button_next = Button(self.button_next_ax, 'Next')
        self.button_jump_prev = Button(self.button_jump_prev_ax, 'Prev x10')
        self.button_jump_next = Button(self.button_jump_next_ax, 'Next x10')

        self.button_prev.on_clicked(self.prev_point)
        self.button_next.on_clicked(self.next_point)
        self.button_jump_prev.on_clicked(self.jump_prev_point)
        self.button_jump_next.on_clicked(self.jump_next_point)

        # Generate and draw UMAP
        self.generate_umap_plot(GAME_ID, RANDOM_SEED)

    def on_canvas_draw(self, event):
        # Refresh dynamic components
        self.refresh_dynamic_components()

    def refresh_dynamic_components(self):
        self.umap_axis.draw_artist(self.highlight_marker)
        self.state_image_axis.draw_artist(self.state_image)

    def generate_umap_plot(self, game, seed):
        print("Loading data...")
        # Load data
        activations = np.load(f'data/{game}/save_activations_{seed}.npy')
        q_values = np.load(f'data/{game}/save_qvalues_{seed}.npy')
        self.value_estimates = np.max(q_values, axis=1)
        self.rewards_data = np.load(f'data/{game}/save_rewards_{seed}.npy')
        self.state_images = np.load(f'data/{game}/save_images_{seed}.npy', allow_pickle=True)
        print("Data successfully loaded.")

        # UMAP embedding
        reducer = umap.UMAP(
            n_neighbors=self.embedding_neighbors,
            n_components=self.embedding_components,
            random_state=seed
        )
        print("Computing UMAP...")
        self.embedded_data = reducer.fit_transform(activations)
        print("UMAP computation complete.")

        self.num_points = self.embedded_data.shape[0]

        # Update scatter plot
        self.umap_axis.cla()
        self.umap_scatter = self.umap_axis.scatter(
            self.embedded_data[:, 0],
            self.embedded_data[:, 1],
            s=5,
            c=self.value_estimates,
            cmap='coolwarm',
            picker=5,
        )
        self.umap_axis.set_xlabel("UMAP Dimension 1", fontsize=12)
        self.umap_axis.set_ylabel("UMAP Dimension 2", fontsize=12)

        # Add color bar
        colorbar_axes = plt.axes([0.04, 0.11, 0.01, 0.78])
        self.color_bar = self.figure.colorbar(self.umap_scatter, cax=colorbar_axes)
        self.color_bar.set_label("Estimated Value", color="white")

        self.figure.canvas.draw()
        self.update_selected_point()

    def prev_point(self, event):
        # Navigate to the previous point
        if self.selected_index > 0:
            self.selected_index -= 1
        self.update_selected_point()

    def next_point(self, event):
        # Navigate to the next point
        if self.selected_index < self.num_points - 1:
            self.selected_index += 1
        self.update_selected_point()

    def jump_prev_point(self, event):
        # Navigate 10 points backward
        self.selected_index = max(0, self.selected_index - 10)
        self.update_selected_point()

    def jump_next_point(self, event):
        # Navigate 10 points forward
        self.selected_index = min(self.num_points - 1, self.selected_index + 10)
        self.update_selected_point()

    def update_selected_point(self):
        # Update image and marker for the selected point
        self.state_image.set_array(self.state_images[self.selected_index])
        self.highlight_marker.set_data(
            [self.embedded_data[self.selected_index, 0]],
            [self.embedded_data[self.selected_index, 1]],
        )
        self.figure.canvas.draw()

    def show(self):
        plt.show(block=True)


# Instantiate and display the visualization tool
visualizer = PacmanVisualizer()
visualizer.show()

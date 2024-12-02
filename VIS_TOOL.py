import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.manifold import TSNE
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

GAME_TITLE = 'Pacman'
GAME_ID = 'MsPacmanNoFrameskip-v4'
RANDOM_SEED = 1732481853


class PacmanVisualizer:
    def __init__(self):
        # Configuration for t-SNE
        self.embedding_perplexity = 60

        # Create figure
        self.figure = plt.figure('Pacman t-SNE View', figsize=(12, 8))
        self.figure.patch.set_facecolor("#f5f5f5")

        # Event listeners for canvas interactions
        self.figure.canvas.mpl_connect('draw_event', self.on_canvas_draw)
        self.figure.canvas.mpl_connect('pick_event', self.on_point_selected)

        # Main t-SNE plot
        self.tsne_axis = plt.subplot2grid((30, 40), (0, 0), rowspan=30, colspan=30)
        self.tsne_axis.set_facecolor("#1e1e1e")
        self.tsne_axis.set_title("t-SNE Visualization of Pacman States", fontsize=16)

        # Placeholder scatter plot
        self.tsne_scatter = self.tsne_axis.scatter([0], [0], picker=True)

        # State image display area
        self.selected_index = 0
        self.state_image_axis = plt.subplot2grid((30, 40), (15, 33), rowspan=15, colspan=7)
        self.state_image = self.state_image_axis.imshow([[[0, 0, 0]]], interpolation='none')
        self.state_image_axis.set_xticklabels([])
        self.state_image_axis.set_yticklabels([])

        # Highlight marker
        self.highlight_marker, = self.tsne_axis.plot(
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

        # Attach button functionality
        self.button_prev.on_clicked(self.prev_point)
        self.button_next.on_clicked(self.next_point)
        self.button_jump_prev.on_clicked(self.jump_prev_point)
        self.button_jump_next.on_clicked(self.jump_next_point)

        # Generate and draw t-SNE plot
        self.generate_tsne_plot(GAME_ID, RANDOM_SEED)

    def generate_tsne_plot(self, game, seed):
        print("Loading data...")
        # Load data
        activations = np.load(f'data/{game}/save_activations_{seed}.npy')
        q_values = np.load(f'data/{game}/save_qvalues_{seed}.npy')
        self.value_estimates = np.max(q_values, axis=1)
        self.rewards_data = np.load(f'data/{game}/save_rewards_{seed}.npy')
        self.state_images = np.load(f'data/{game}/save_images_{seed}.npy', allow_pickle=True)
        print("Data successfully loaded.")

        # Perform t-SNE embedding
        tsne = TSNE(
            n_components=2,
            perplexity=self.embedding_perplexity,
            verbose=1,
            random_state=seed,
            method='barnes_hut'
        )
        print("Computing t-SNE...")
        self.embedded_data = tsne.fit_transform(activations)
        print("t-SNE computation complete.")

        self.num_points = self.embedded_data.shape[0]

        # Update scatter plot
        self.tsne_axis.cla()
        self.tsne_scatter = self.tsne_axis.scatter(
            self.embedded_data[:, 0],
            self.embedded_data[:, 1],
            s=5,
            c=self.value_estimates,
            cmap='coolwarm',
            picker=True,  # Enable point picking
        )
        self.tsne_axis.set_xlabel("t-SNE Dimension 1", fontsize=12)
        self.tsne_axis.set_ylabel("t-SNE Dimension 2", fontsize=12)

        # Add color bar
        colorbar_axes = plt.axes([0.04, 0.11, 0.01, 0.78])
        self.color_bar = self.figure.colorbar(self.tsne_scatter, cax=colorbar_axes)
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

    def on_point_selected(self, event):
        # Handle mouse click on scatter plot points
        if event.artist == self.tsne_scatter:
            ind = event.ind[0]
            self.selected_index = ind
            self.update_selected_point()

    def update_selected_point(self):
        # Update image and marker for the selected point
        self.state_image.set_array(self.state_images[self.selected_index])
        self.highlight_marker.set_data(
            [self.embedded_data[self.selected_index, 0]],
            [self.embedded_data[self.selected_index, 1]],
        )
        self.figure.canvas.draw()

    def on_canvas_draw(self, event):
        # Refresh dynamic components
        self.tsne_axis.draw_artist(self.highlight_marker)
        self.state_image_axis.draw_artist(self.state_image)

    def show(self):
        plt.show(block=True)


# Instantiate and display the visualization tool
visualizer = PacmanVisualizer()
visualizer.show()

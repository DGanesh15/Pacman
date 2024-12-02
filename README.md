# Pacman
# Graying the Black Box: Understanding DQNs - High Dimensional Data Analysis Visualization Project

This project implements t-SNE and UMAP-based visualizations of high-dimensional activation data from Deep Q-Networks (DQNs), inspired by the research paper *Graying the Black Box: Understanding DQNs*. The code enables understanding how DQNs make decisions by mapping their internal activations to a 2D space, making it easier to interpret relationships between states, Q-values, and game states.

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure that you have the following installed on your system:
- **Conda**: For creating and managing the Python environment.
- **Python 3.8**: This project is tested with Python 3.8.

### **2. Create a Conda Environment**
Run the following commands to create and activate a Conda environment for this project:

```bash
conda create -n dqn-visualization python=3.8 -y
conda activate dqn-visualization
```

### **3. Install Required Packages**
Install the necessary Python libraries using `pip`:

```bash
pip install numpy matplotlib scikit-learn cuml gymnasium stable-baselines3
```

---

## **Project Overview**

### **Key Functionalities**
- **t-SNE Visualization**:
  - Dimensionality reduction using t-SNE to project high-dimensional DQN activations to 2D space.
  - Highlights the relationship between Q-values, states, and rewards.
  
- **UMAP Visualization**:
  - Similar to t-SNE, but using UMAP for dimensionality reduction.
  - Provides a faster and scalable alternative for larger datasets.

- **Interactive Controls**:
  - Navigate through game states using buttons.
  - You can add an extra bit of code to adjust visualization parameters like perplexity using sliders dynamically, which I haven't done.
  - Recompute analysis for different random seeds and games.

---

## **How to Run the Project**

### **1. Prepare Data**
The data for this project should include:
- Activation data: `save_activations_<seed>.npy`
- Q-values: `save_qvalues_<seed>.npy`
- Rewards: `save_rewards_<seed>.npy`
- State images: `save_images_<seed>.npy`

Organize the data for each game in the following folder structure:
```
data/
    MsPacmanNoFrameskip-v4/
        save_activations_<seed>.npy
        save_qvalues_<seed>.npy
        save_rewards_<seed>.npy
        save_images_<seed>.npy
```
You can also generate the data using 
```bash
python generate_data.py
```
and you will get data in required format but remember to change the game seed and ID for future HDDA tasks

### **2. Run the Visualization**
To start the t-SNE/UMAP visualization for Pacman, run the appropriate script:

```bash
python VIS_TOOL.py
```

OR

```bash
python visualization_umap.py
```

Follow the interactive controls in the matplotlib window:
- **Previous/Next**: Navigate one frame backward/forward.
- **PREVx10/NEXTx10**: Jump 10 frames backward/forward.

---

## **Adapting for Other Atari2600 Games**

### **1. Data Preparation**
1. **Replace Game Data**:
   Collect activation, Q-values, rewards, and state images for the new game (e.g., Breakout) following the same format.
   
2. **Update Folder Structure**:
   Create a new folder inside `data/` named after the game’s Gym environment ID, such as `BreakoutNoFrameskip-v4`.

3. **Adjust Data Files**:
   Ensure the data files are named using the same convention (`save_activations_<seed>.npy`, etc.).

---

### **2. Update the Code**

1. **Change the Game Name**:
   Update the `GAME_ID` and `GAME_TITLE` variables in the script to the new game’s environment ID and title:
   ```python
   GAME_TITLE = 'Breakout'
   GAME_ID = 'BreakoutNoFrameskip-v4'
   ```

2. **Use a Different Seed (Optional)**:
   Update the `RANDOM_SEED` variable if desired:
   ```python
   RANDOM_SEED = <new_seed_value>
   ```

3. **Recompute Analysis**:
   The buttons in the script support dynamic recomputation for new games. After updating `GAME_ID`, you can directly use the interactive controls.

---

## **Future Enhancements**
- **Batch Processing**: Add support to process multiple games in one run and visualize comparisons.
- **Deep Analysis**: Include additional insights such as clustering of states, distance metrics, and temporal relationships between states.
- **Web-based Interface**: Migrate the visualization to a web-based interface using tools like Plotly Dash for better interactivity.

---

## **References**
- **Paper**: Graying the Black Box: Understanding DQNs
- **Libraries Used**:
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [Scikit-learn](https://scikit-learn.org/)
  - [CUML](https://rapids.ai/)
  - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
  - [Gymnasium](https://gymnasium.farama.org/)

This project is a direct implementation of the concepts from the research paper, making it accessible for researchers and developers interested in reinforcement learning interpretability.

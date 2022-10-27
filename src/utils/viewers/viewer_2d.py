
import matplotlib.pyplot as plt

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(voxel):
    remove_keymap_conflicts({'j', 'k'})
    if len(voxel.shape) == 4:
        voxel = voxel.squeeze()
    fig, ax = plt.subplots()
    ax.voxel = voxel
    ax.index = voxel.shape[0] // 2
    ax.imshow(voxel[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    voxel = ax.voxel
    ax.index = (ax.index - 1) % voxel.shape[0]  # wrap around using %
    ax.images[0].set_array(voxel[ax.index])

def next_slice(ax):
    voxel = ax.voxel
    ax.index = (ax.index + 1) % voxel.shape[0]
    ax.images[0].set_array(voxel[ax.index])

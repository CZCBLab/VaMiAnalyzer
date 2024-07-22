import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, morphology, color
import networkx as nx
from tqdm import tqdm
from collections import deque, defaultdict
from matplotlib.colors import ListedColormap
import csv
import argparse

directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
directions2 = [[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
               [-2, 1], [2, 1],
               [-2, 0], [2, 0],
               [-2, -1], [2, -1],
               [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2]]

def adjust_mean_std(image, target_mean, target_std):
    mean, std = cv2.meanStdDev(image)
    mean = mean[0][0]
    std = std[0][0]
    
    gain = target_std / std
    bias = target_mean - gain * mean
    
    adjusted_image = image.astype(np.float32)
    adjusted_image = (adjusted_image - mean) * gain + target_mean
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    
    return adjusted_image

def binary_image(window_size, step_size, imagefile, area_threshold, std_threshold, margin):
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    window_size = (window_size, window_size) 
    binary_image = np.zeros_like(image)

    for y in range(margin, image.shape[0] - window_size[1] - margin + 1, step_size):
        for x in range(margin, image.shape[1] - window_size[0] - margin + 1, step_size):

            window = image[y:y + window_size[1], x:x + window_size[0]]

            std_dev = np.std(window)

            if std_dev > std_threshold:
                binary_image[y:y + window_size[1], x:x + window_size[0]] = 1

    labels = measure.label(binary_image, connectivity=1)
    label_counts = np.bincount(labels.flat)[1:]

    main_labels = np.where(label_counts > area_threshold)[0] + 1
    main_region_image = np.isin(labels, main_labels).astype(int)

    main_region_image = 1-main_region_image
    labels = measure.label(main_region_image, connectivity=1)
    label_counts = np.bincount(labels.flat)
    for label, count in enumerate(label_counts):
        if count<area_threshold:
            labels[labels==label]=0
    main_region_image = labels>0
    main_region_image = 1-main_region_image

    kernel = morphology.square(20)
    for i in range(1):
        main_region_image = morphology.opening(main_region_image, kernel)

    kernel = morphology.square(20)
    for i in range(1):
        main_region_image = morphology.closing(main_region_image, kernel)

    labels = measure.label(main_region_image, connectivity=1)
    label_counts = np.bincount(labels.flat)[1:]
    main_labels = np.where(label_counts > area_threshold)[0] + 1
    main_region_image = np.isin(labels, main_labels).astype(int)

    main_region_image = 1 - main_region_image
    labels = measure.label(main_region_image, connectivity=1)
    label_counts = np.bincount(labels.flat)
    for label, count in enumerate(label_counts):
        if count < area_threshold:
            labels[labels == label] = 0
    main_region_image = labels > 0
    main_region_image = 1 - main_region_image

    kernel = morphology.square(40)
    for i in range(1):
        main_region_image = morphology.closing(main_region_image, kernel)

    return main_region_image

def mask_correct(G, mask):
    def is_within_circle(center_x, center_y, test_x, test_y, radius):
        return (test_x - center_x) ** 2 + (test_y - center_y) ** 2 <= radius ** 2
    
    height, width = mask.shape
    region = np.zeros((height, width), dtype=np.uint8)
    for node in G.nodes():
        center_x, center_y = node[1], node[0]
        xmin = max(0, center_x - 200)
        xmax = min(mask.shape[1], center_x + 200)
        ymin = max(0, center_y - 200)
        ymax = min(mask.shape[0], center_y + 200)

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                if is_within_circle(center_x, center_y, x, y, 200):
                    region[y, x] = 1

    for y in range(height):
        for x in range(width):
            if (mask[y, x] == 1 and region[y, x] == 0):
                mask[y, x] = 0

    return mask

def find_endpoints(skel):
    endpoint_kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)
    convolution_result = cv2.filter2D(skel.astype(np.uint8), -1, endpoint_kernel)
    endpoints = (convolution_result == 11)
    return endpoints

def find_bifurcation_points(skel):
    bifurcation_kernel = np.array([[1, 1, 1],
                                   [1, 10, 1],
                                   [1, 1, 1]], dtype=np.uint8)
    convolution_result = cv2.filter2D(skel.astype(np.uint8), -1, bifurcation_kernel)
    bifurcation_points = (convolution_result >= 13)
    return bifurcation_points

def merge_close_nodes(skel, graph, threshold, image):
    merged_graph = graph.copy()
    nodes_to_remove = []
    groupdict = {}
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            if graph[node][neighbor].get('weight', 1) < threshold:
                if node in nodes_to_remove:
                    if neighbor in groupdict.keys():
                        continue
                    elif neighbor in nodes_to_remove:
                        continue
                    else:
                        for key, values in groupdict.items():
                            if node in values:
                                merged_graph = nx.contracted_nodes(merged_graph, key, neighbor, self_loops=False)
                                nodes_to_remove.append(neighbor)
                                groupdict[key].append(neighbor)
                        continue

                if neighbor in nodes_to_remove:
                    for key, values in groupdict.items():
                        if neighbor in values:
                            break
                    merged_graph = nx.contracted_nodes(merged_graph, key, node, self_loops=False)
                    nodes_to_remove.append(node)
                    groupdict[key].append(node)
                    break

                merged_graph = nx.contracted_nodes(merged_graph, node, neighbor, self_loops=False)
                nodes_to_remove.append(neighbor)
                if node in groupdict:
                    groupdict[node].append(neighbor)
                else:
                    groupdict[node] = [neighbor]

    merged_graph.remove_nodes_from(nodes_to_remove)
    return merged_graph, nodes_to_remove, groupdict

def merge_close_nodes2(G, merge_threshold, pathdict, image):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    to_merge = {}
    for u, v, data in G.edges(data=True):
        if data['weight'] < merge_threshold:
            if u not in to_merge and v not in to_merge:
                to_merge[v] = u

    for v, u in to_merge.items():
        for x in G.neighbors(v):
            if x != u and x not in to_merge:
                new_path = pathdict.get((u, v), []) + pathdict.get((v, x), [])
                new_weight = len(new_path)
                if H.has_edge(u, x):
                    # if H[u][x]['weight'] > new_weight:
                    H[u][x]['weight'] = new_weight
                    pathdict[(u, x)] = new_path
                    pathdict[(x, u)] = new_path
                else:
                    H.add_edge(u, x, weight=new_weight)
                    pathdict[(u, x)] = new_path
                    pathdict[(x, u)] = new_path

    for v in to_merge:
        H.remove_node(v)

    for u, v, data in G.edges(data=True):
        if u not in to_merge and v not in to_merge and not H.has_edge(u, v):
            H.add_edge(u, v, weight=data['weight'])

    return H

def get_neighbors(skel, point, directions, im_height, im_width):
    x, y = point
    neighbors = []
    for dx, dy in directions:
        if x + dx < im_height and y + dy < im_width:
            if skel[x + dx, y + dy]:
                neighbors.append((x + dx, y + dy))
    return neighbors

def build_graph(skeleton, endpoint_coords, bifurcation_coords):
    visited = np.zeros_like(skeleton, dtype=bool)
    G = nx.Graph()
    pathdict = defaultdict(list)
    for x in range(skeleton.shape[0]):
        for y in range(skeleton.shape[1]):
            if skeleton[x, y] and not visited[x, y]:
                dfs(skeleton, x, y, visited, G, pathdict, endpoint_coords, bifurcation_coords)
    return G, pathdict

def dfs(skeleton, start_x, start_y, visited, G, pathdict, endpoint_coords, bifurcation_coords, parent=None, path=None, path_length=0):
    stack = [(start_x, start_y, parent, path if path is not None else [], path_length)]
    while stack:
        x, y, parent, path, path_length = stack.pop()
        visited[x, y] = True
        cur = (x, y)
        path.append(cur)

        if cur in endpoint_coords or cur in bifurcation_coords:
            G.add_node(cur)
            if parent is not None:
                G.add_edge(parent, cur, weight=path_length)
                pathdict[(parent, cur)] = list(path)
                pathdict[(cur, parent)] = list(path[::-1])
            parent = cur
            path = []
            path_length = 0

        neighbors = get_neighbors(skeleton, [x, y], directions1, skeleton.shape[0], skeleton.shape[1])
        for nx, ny in reversed(neighbors): 
            if skeleton[nx, ny] and not visited[nx, ny]:
                stack.append((nx, ny, parent, list(path), path_length + 1))
            elif skeleton[nx, ny] and visited[nx, ny] and len(path) > 2 and ((nx, ny) in endpoint_coords or (nx, ny) in bifurcation_coords):
                G.add_node((nx, ny))
                if parent is not None:
                    G.add_edge(parent, (nx, ny), weight=path_length)
                    pathdict[(parent, (nx, ny))] = list(path)
                    pathdict[((nx, ny), parent)] = list(path[::-1])
                parent = (nx, ny)
                path = []
                path_length = 0

def rebuild_skeleton(skeleton, merged_graph, pathdict, groupdict):
    newskel = []
    for edge in merged_graph.edges():
        if (edge[0], edge[1]) in pathdict or (edge[1], edge[0]) in pathdict:
            newskel = newskel + pathdict[(edge[0], edge[1])] + pathdict[(edge[1], edge[0])]
        else:
            l_0 = [edge[0]]
            l_1 = [edge[1]]
            l_0 = l_0 + groupdict.get(edge[0],[])
            l_1 = l_1 + groupdict.get(edge[1], [])

            pathdict[(edge[0], edge[0])] = []
            pathdict[(edge[1], edge[1])] = []
            for i in range(len(l_0)):
                for j in range(len(l_1)):
                    if (l_0[i] == (1317, 572) or l_1[j] == (1317, 572)) and (l_0[i] == (1277, 521) or l_1[j] == (1277, 521)):
                        print('hello')
                    if (l_0[i], l_1[j]) in pathdict:
                        if not (edge[0], l_0[i]) in pathdict.keys():
                            pathdict[(edge[0], l_0[i])] = pathdict[(edge[0], l_0[i])]
                        if not (l_1[j], edge[1]) in pathdict.keys():
                            pathdict[(l_1[j], edge[1])] = pathdict[(l_1[j], edge[1])]


                        newskel = newskel + pathdict[(edge[0], l_0[i])] + pathdict[(l_0[i], l_1[j])] + pathdict[(l_1[j], l_0[i])] + pathdict[(l_1[j], edge[1])] + pathdict[(l_0[i], edge[0])] + pathdict[(edge[1], l_1[j])]
    
    newskelimage = np.zeros(skeleton.shape)
    rows, cols = zip(*newskel)
    newskelimage[rows, cols] = True
    
    return newskelimage

def findBranchPoints(G, merge_threshold):
    branch_points = [node for node, degree in dict(G.degree()).items() if degree > 2]

    to_merge = []
    for u, v, data in G.edges(data=True):
        if data['weight'] < merge_threshold:
            if u not in to_merge and v not in to_merge:
                to_merge.append(v)

    filtered_btanch_points = [item for item in branch_points if item not in to_merge]
    
    return filtered_btanch_points

def remove_small_branches(G, length_threshold, image, radius):
    def is_within_circle(center_x, center_y, test_x, test_y, radius):
        return (test_x - center_x) ** 2 + (test_y - center_y) ** 2 <= radius ** 2

    leaf_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if (u in leaf_nodes or v in leaf_nodes) and d['weight'] < length_threshold]
    
    while leaf_nodes and filtered_edges:
        G.remove_edges_from(filtered_edges)
        nodes_to_remove = [node for node in leaf_nodes if G.degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
        leaf_nodes = [node for node, degree in dict(G.degree()).items() if degree == 1]
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if (u in leaf_nodes or v in leaf_nodes) and d['weight'] < length_threshold]
        
    nodes_to_remove = []
    for node in G.nodes():
        center_x, center_y = node[1], node[0]
        xmin = max(0, center_x - radius)
        xmax = min(image.shape[1], center_x + radius + 1)
        ymin = max(0, center_y - radius)
        ymax = min(image.shape[0], center_y + radius + 1)

        if xmin < xmax and ymin < ymax:
            region = np.ones((ymax - ymin, xmax - xmin), dtype=image.dtype)
        else:
            region = np.array([])
        
        width = xmax - xmin
        height = ymax - ymin
        max_circle_radius = min(width / 2, height / 2)

        if (max_circle_radius >= radius):
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    if is_within_circle(center_x, center_y, x, y, radius):
                        region[y - ymin, x - xmin] = image[y, x]

            if np.all(region == 1):
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if G.degree(neighbor) == 1:
                        nodes_to_remove.append(neighbor)

    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                            if (u in nodes_to_remove or v in nodes_to_remove)]
    while nodes_to_remove and edges_to_remove:
        G.remove_edges_from(edges_to_remove)
        nodes_to_remove = [node for node in nodes_to_remove if G.degree(node) == 0]
        G.remove_nodes_from(nodes_to_remove)
        nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 1]
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                        if (u in nodes_to_remove or v in nodes_to_remove) and d['weight'] < length_threshold]
        
    return G

def count_edges(G, length_threshold):
    long_edges_count = sum(1 for u, v, d in G.edges(data=True) if d['weight'] > length_threshold)
    return long_edges_count

def filter_loop(loops, branch_points):
    valid_loops = []

    for loop in loops:
        branch_count = sum(1 for node in loop if node in branch_points)

        if branch_count >= 2:
            valid_loops.append(loop)

    return valid_loops

def draw_result(skeleton, image, mask_image, branch_points, loops, filename, output_dir):
    kernel = morphology.square(10)
    skeleton = skeleton>0
    skeleton_dilated = morphology.dilation(skeleton, kernel)
    skeleton_dilated = skeleton_dilated>0
    image[skeleton_dilated] = [255, 255, 255]

    plt.figure()
    plt.imshow(image)

    if branch_points:
        x_coords, y_coords = zip(*[(node[0], node[1]) for node in branch_points])
        plt.scatter(y_coords, x_coords, color='yellow', s=20, edgecolors='black')

    if loops: 
        for index, loop in enumerate(loops):
            x_coords, y_coords = zip(*loop)
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            plt.text(centroid_y, centroid_x, str(index + 1), 
            color='orange', fontsize=12,
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(facecolor='none', edgecolor='orange', boxstyle='circle,pad=0.3'))

    plt.axis('off')
    plt.savefig(f"{output_dir}/result_{filename}")
    # plt.show()
    plt.close()
    print('done!')

def save_results_to_csv(results, csv_path):
    sorted_results = sorted(results, key=lambda x: x['filename'])

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'num_loops', 'num_branch_points', 'num_tubulars']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)
    
def process_images(input_path, output_dir, mean, std, window_size, step_size, merge_threshold, min_tubular_length, min_end_tubular_length, area_threshold, std_threshold, radius, margin):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = []
    if os.path.isfile(input_path) and input_path.endswith('.tif'):
        dir, file = os.path.split(input_path)
        filenames = [file]
    else:
        dir = input_path
        filenames = [f for f in os.listdir(input_path) if f.endswith(".tif")]

    results = []
    for filename in tqdm(filenames, desc='Processing Images'):
        file_path = os.path.join(dir, filename)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue

        image = adjust_mean_std(image, mean, std)

        if image.ndim != 3 or image.shape[2] != 3:
            image = color.gray2rgb(image)
        
        mask_image= binary_image(window_size, step_size, file_path, area_threshold, std_threshold, margin)
        skeleton = morphology.skeletonize(mask_image)

        endpoints = find_endpoints(skeleton)
        bifurcation_points = find_bifurcation_points(skeleton)

        endpoint_coords = list(zip(*np.where(endpoints)))
        bifurcation_coords = list(zip(*np.where(bifurcation_points)))
        G, pathdict = build_graph(skeleton, endpoint_coords, bifurcation_coords)
        merged_graph, removed_node, groupdict = merge_close_nodes(skeleton, G, 5, image)
        pruned_graph = remove_small_branches(merged_graph, min_end_tubular_length, mask_image, radius)
        new_skel = rebuild_skeleton(skeleton, pruned_graph, pathdict, groupdict)
        new_skel = new_skel.astype(np.uint8)
        branch_points = findBranchPoints(pruned_graph, merge_threshold)
        loops = nx.minimum_cycle_basis(pruned_graph)
        valid_loops = filter_loop(loops, branch_points)
        num_tubulars = count_edges(pruned_graph, min_tubular_length)

        draw_result(new_skel, image, mask_image, branch_points, valid_loops, filename, output_dir)

        results.append({
            'filename': filename,
            'num_loops': len(valid_loops),
            'num_branch_points': len(branch_points),
            'num_tubulars': num_tubulars
        })

    csv_path = os.path.join(output_dir, 'results.csv')
    save_results_to_csv(results, csv_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process images and analyze tubular structures.')
    parser.add_argument('input_path', type=str, help='Path to the input file or directory.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')
    parser.add_argument('--mean', type=int, default=95, help='Mean value for adjusting image.')
    parser.add_argument('--std', type=int, default=14, help='Standard deviation value for adjusting image.')
    parser.add_argument('--window_size', type=int, default=20, help='Window size for binary image processing.')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for binary image processing.')
    parser.add_argument('--merge_threshold', type=int, default=50, help='Threshold for merging close nodes in the graph.')
    parser.add_argument('--min_tubular_length', type=int, default=120, help='Minimum length of tubular structures to be considered for counting the number of tubes.')
    parser.add_argument('--min_end_tubular_length', type=int, default=200, help='Minimum length of end tubular structures for pruning.')
    parser.add_argument('--area_threshold', type=int, default=3000, help='Minimum area for regions to be considered in binary image.')
    parser.add_argument('--std_threshold', type=int, default=8, help='Standard deviation threshold for binary image creation.')
    parser.add_argument('--radius', type=int, default=150, help='Radius for local region processing in small branch removal.')
    parser.add_argument('--margin', type=int, default=5, help='Margin to be excluded from binary image creation.')

    args = parser.parse_args()
    process_images(args.input_path, args.output_dir, args.mean, args.std, args.window_size, args.step_size, args.merge_threshold, args.min_tubular_length, args.min_end_tubular_length, args.area_threshold, args.std_threshold, args.radius, args.margin)
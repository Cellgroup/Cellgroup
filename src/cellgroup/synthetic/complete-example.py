import numpy as np
import matplotlib.pyplot as plt
from cellgroup.synthetic import Space, Nucleus, Cluster, Sample, NucleusFluorophoreDistribution

def run_basic_simulation():
    """Run a basic cell simulation and visualization."""
    # 1. Create a simulation space
    space = Space(
        size=(512, 512),
        scale=(0.5, 0.5)
    )
    
    # 2. Set up simulation parameters
    n_clusters = 3
    nuclei_per_cluster_range = (10, 15)
    nuclei_semi_axes_range = (8, 15)
    
    # 3. Generate cluster parameters
    cluster_centers = []
    cluster_radii = []
    min_distance = 150
    
    # Generate random cluster centers with minimum separation
    while len(cluster_centers) < n_clusters:
        center = (
            np.random.uniform(100, 412),
            np.random.uniform(100, 412)
        )
        
        # Check if this center is far enough from existing centers
        too_close = False
        for existing_center in cluster_centers:
            dist = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(center, existing_center)))
            if dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            cluster_centers.append(center)
            # Random radius for each cluster
            radius = np.random.uniform(40, 80)
            cluster_radii.append((radius, radius))
    
    # 4. Create the sample with clusters
    sample = Sample.create_random_sample(
        space=space,
        time=0,
        n_clusters=n_clusters,
        cluster_centers=cluster_centers,
        cluster_radii=cluster_radii,
        nuclei_per_cluster_range=nuclei_per_cluster_range,
        nuclei_semi_axes_range=nuclei_semi_axes_range,
        # Additional parameters for cluster behavior
        cluster_interaction_range=100.0,
        cluster_merge_threshold=30.0
    )
    
    # 5. Run simulation for multiple timesteps
    n_timesteps = 10
    images = []
    metrics_history = []
    
    for t in range(n_timesteps):
        print(f"Simulating timestep {t+1}/{n_timesteps}")
        
        # Update all clusters in the sample
        sample.update()
        
        # Render the current state
        image = sample.render()
        images.append(image)
        
        # Collect metrics
        metrics = sample.get_cluster_metrics()
        metrics_history.append(metrics)
        
        # Print basic statistics
        print(f"  Number of clusters: {sample.count}")
        print(f"  Total number of nuclei: {sample.total_nuclei}")
        print(f"  Nuclei per cluster: {sample.nuclei_count}")
        print()
    
    # 6. Visualize results
    # Create a figure for the time series
    plt.figure(figsize=(15, 8))
    
    # Show selected frames
    frames_to_show = min(5, n_timesteps)
    indices = np.linspace(0, n_timesteps-1, frames_to_show, dtype=int)
    
    for i, idx in enumerate(indices):
        plt.subplot(1, frames_to_show, i+1)
        plt.imshow(images[idx], cmap='viridis')
        plt.title(f"Timestep {idx+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("cell_simulation_frames.png")
    
    # Plot metrics
    plt.figure(figsize=(15, 5))
    
    # Total nuclei over time
    plt.subplot(1, 3, 1)
    total_nuclei = [m['total_nuclei'] for m in metrics_history]
    plt.plot(range(1, n_timesteps+1), total_nuclei, 'o-')
    plt.title("Total Nuclei Count")
    plt.xlabel("Timestep")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Number of clusters over time
    plt.subplot(1, 3, 2)
    n_clusters = [m['n_clusters'] for m in metrics_history]
    plt.plot(range(1, n_timesteps+1), n_clusters, 'o-')
    plt.title("Number of Clusters")
    plt.xlabel("Timestep")
    plt.ylabel("Count")
    plt.grid(True)
    
    # Mean nuclei per cluster
    plt.subplot(1, 3, 3)
    mean_cluster_size = [m['mean_cluster_size'] for m in metrics_history]
    plt.plot(range(1, n_timesteps+1), mean_cluster_size, 'o-')
    plt.title("Mean Nuclei per Cluster")
    plt.xlabel("Timestep")
    plt.ylabel("Count")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cell_simulation_metrics.png")
    
    # Show both figures
    plt.show()
    
    return sample, images, metrics_history

def run_fluorophore_simulation():
    """Run a simulation with fluorophore distributions."""
    # 1. Create a simulation space
    space = Space(
        size=(256, 256),
        scale=(0.5, 0.5)
    )
    
    # 2. Generate an image with multiple nuclei and their fluorescence
    distribution_types = ["gaussian", "ring", "uniform"]
    images = {}
    
    for dist_type in distribution_types:
        # Create empty image
        image = np.zeros(space.size)
        
        # Generate 10 random nuclei
        for i in range(10):
            # Random position
            x = np.random.uniform(50, space.size[1] - 50)
            y = np.random.uniform(50, space.size[0] - 50)
            
            # Random size
            major = np.random.uniform(15, 25)
            minor = major * np.random.uniform(0.7, 1.0)
            
            # Random orientation
            angle = np.random.uniform(0, 360)
            
            # Create nucleus
            nucleus = Nucleus(
                idx=i,
                time=0,
                space=space,
                centroid=(x, y),
                semi_axes=(major, minor),
                angle_x=angle,
                raw_int_density=1000
            )
            
            # Create fluorophore distribution
            fp_dist = NucleusFluorophoreDistribution(
                nucleus=nucleus,
                distribution_type=dist_type,
                fluorophore_density=np.zeros(space.size),
                intensity_center=1.0,
                intensity_edge=0.2,
                noise_std=0.05,
                background_level=0.02
            )
            
            # Add to image
            image += fp_dist.render(space)
        
        images[dist_type] = image
    
    # 3. Visualize the results
    plt.figure(figsize=(15, 5))
    
    for i, (dist_type, image) in enumerate(images.items()):
        plt.subplot(1, 3, i+1)
        plt.imshow(image, cmap='viridis')
        plt.title(f"{dist_type.capitalize()} Distribution")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("fluorophore_distributions.png")
    plt.show()
    
    return images

if __name__ == "__main__":
    print("=== Running Basic Cell Simulation ===")
    sample, images, metrics = run_basic_simulation()
    
    print("\n\n=== Running Fluorophore Distribution Simulation ===")
    fp_images = run_fluorophore_simulation()
    
    print("\nSimulations complete. Results saved as PNG files.")

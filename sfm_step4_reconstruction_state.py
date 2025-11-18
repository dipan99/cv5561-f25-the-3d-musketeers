import numpy as np
import pickle


class ReconstructionState:
    """
    Data structure to track the state of the 3D reconstruction
    
    Maintains:
    - Camera poses for each image
    - 3D points
    - Tracks: mapping of which 3D points are seen in which images
    """
    
    def __init__(self, K):
        """
        Initialize reconstruction state
        
        Args:
            K: Camera intrinsic matrix (3x3)
        """
        self.K = K
        
        # Camera poses: {image_idx: {'R': R, 't': t, 'registered': bool}}
        self.cameras = {}
        
        # 3D points: list of 3D coordinates
        self.points_3d = []
        
        # Tracks: list of observations for each 3D point
        # Each track is a dict: {image_idx: 2d_point_coords}
        self.tracks = []
        
        # Metadata
        self.image_paths = []
        self.num_images = 0
    
    def add_camera(self, img_idx, R, t, registered=True):
        """
        Add or update a camera pose
        
        Args:
            img_idx: Image index
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            registered: Whether this camera is fully registered
        """
        self.cameras[img_idx] = {
            'R': R.copy(),
            't': t.copy(),
            'registered': registered
        }
        print(f"[state] Added camera {img_idx} (registered={registered})")
    
    def add_point(self, point_3d, observations):
        """
        Add a 3D point with its observations
        
        Args:
            point_3d: 3D coordinates (3,) or (3,1)
            observations: dict {image_idx: 2d_point (2,)}
        """
        # Ensure point is (3,) shape
        if point_3d.shape == (3, 1):
            point_3d = point_3d.ravel()
        
        self.points_3d.append(point_3d.copy())
        self.tracks.append(observations.copy())
    
    def add_points_batch(self, points_3d, observations_list):
        """
        Add multiple 3D points with their observations
        
        Args:
            points_3d: (N, 3) array of 3D points
            observations_list: list of N dicts, each {image_idx: 2d_point}
        """
        for point, obs in zip(points_3d, observations_list):
            self.add_point(point, obs)
        
        print(f"[state] Added {len(points_3d)} 3D points")
    
    def get_camera(self, img_idx):
        """Get camera pose for an image"""
        if img_idx not in self.cameras:
            return None
        return self.cameras[img_idx]
    
    def is_camera_registered(self, img_idx):
        """Check if a camera is registered"""
        if img_idx not in self.cameras:
            return False
        return self.cameras[img_idx]['registered']
    
    def get_registered_cameras(self):
        """Get list of registered camera indices"""
        return [idx for idx, cam in self.cameras.items() if cam['registered']]
    
    def get_projection_matrix(self, img_idx):
        """
        Get projection matrix P = K[R|t] for a camera
        
        Args:
            img_idx: Image index
            
        Returns:
            P: Projection matrix (3x4) or None if camera not found
        """
        cam = self.get_camera(img_idx)
        if cam is None:
            return None
        
        R = cam['R']
        t = cam['t']
        P = self.K @ np.hstack([R, t])
        return P
    
    def get_points_observed_by_camera(self, img_idx):
        """
        Get all 3D points observed by a specific camera
        
        Args:
            img_idx: Image index
            
        Returns:
            point_indices: List of 3D point indices
            points_2d: Corresponding 2D observations
        """
        point_indices = []
        points_2d = []
        
        for point_idx, track in enumerate(self.tracks):
            if img_idx in track:
                point_indices.append(point_idx)
                points_2d.append(track[img_idx])
        
        return point_indices, np.array(points_2d) if points_2d else np.array([])
    
    def get_cameras_observing_point(self, point_idx):
        """
        Get all cameras that observe a specific 3D point
        
        Args:
            point_idx: Index of 3D point
            
        Returns:
            camera_indices: List of image indices
            points_2d: Corresponding 2D observations
        """
        if point_idx >= len(self.tracks):
            return [], np.array([])
        
        track = self.tracks[point_idx]
        camera_indices = list(track.keys())
        points_2d = [track[idx] for idx in camera_indices]
        
        return camera_indices, np.array(points_2d)
    
    def summary(self):
        """Print summary of reconstruction state"""
        print("\n" + "="*60)
        print("RECONSTRUCTION STATE SUMMARY")
        print("="*60)
        print(f"Total images: {self.num_images}")
        print(f"Registered cameras: {len(self.get_registered_cameras())}")
        print(f"Total 3D points: {len(self.points_3d)}")
        
        if self.cameras:
            print(f"\nRegistered camera indices: {sorted(self.get_registered_cameras())}")
        
        if self.points_3d:
            points_array = np.array(self.points_3d)
            print(f"\n3D Point Cloud:")
            print(f"  X range: [{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}]")
            print(f"  Y range: [{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}]")
            print(f"  Z range: [{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}]")
        
        # Track statistics
        if self.tracks:
            track_lengths = [len(track) for track in self.tracks]
            print(f"\nTrack statistics:")
            print(f"  Min observations per point: {min(track_lengths)}")
            print(f"  Max observations per point: {max(track_lengths)}")
            print(f"  Avg observations per point: {np.mean(track_lengths):.2f}")
        
        print("="*60)
    
    def save(self, filepath):
        """Save reconstruction state to file"""
        data = {
            'K': self.K,
            'cameras': self.cameras,
            'points_3d': self.points_3d,
            'tracks': self.tracks,
            'image_paths': self.image_paths,
            'num_images': self.num_images
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[state] Saved reconstruction state to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load reconstruction state from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        state = cls(data['K'])
        state.cameras = data['cameras']
        state.points_3d = data['points_3d']
        state.tracks = data['tracks']
        state.image_paths = data['image_paths']
        state.num_images = data['num_images']
        
        print(f"[state] Loaded reconstruction state from {filepath}")
        return state


def initialize_reconstruction_state_from_bootstrap():
    """
    Initialize reconstruction state from the bootstrap data (Steps 1-5)
    
    Returns:
        state: ReconstructionState object
    """
    print("\n" + "="*60)
    print("STEP 6: Setting Up Reconstruction Data Structure")
    print("="*60)
    
    # Load bootstrap data
    print("\n[init] Loading bootstrap data...")
    bootstrap = np.load("./reconstruction_initial.npz", allow_pickle=True)
    
    K = bootstrap['K']
    R0 = bootstrap['R0']
    t0 = bootstrap['t0']
    R1 = bootstrap['R1']
    t1 = bootstrap['t1']
    points_3d = bootstrap['points_3d']
    obs_2d_img0 = bootstrap['obs_2d_img0']
    obs_2d_img1 = bootstrap['obs_2d_img1']
    seed_pair = tuple(bootstrap['seed_pair'])
    image_paths = bootstrap['image_paths']
    
    print(f"[init] Seed pair: images {seed_pair[0]} and {seed_pair[1]}")
    print(f"[init] {len(points_3d)} 3D points from bootstrap")
    
    # Create reconstruction state
    state = ReconstructionState(K)
    state.image_paths = list(image_paths)
    state.num_images = len(image_paths)
    
    # Add the two cameras from bootstrap
    state.add_camera(seed_pair[0], R0, t0, registered=True)
    state.add_camera(seed_pair[1], R1, t1, registered=True)
    
    # Add 3D points with their observations
    print(f"\n[init] Adding {len(points_3d)} 3D points with tracks...")
    observations_list = []
    for i in range(len(points_3d)):
        obs = {
            seed_pair[0]: obs_2d_img0[i],
            seed_pair[1]: obs_2d_img1[i]
        }
        observations_list.append(obs)
    
    state.add_points_batch(points_3d, observations_list)
    
    # Print summary
    state.summary()
    
    # Save state
    state.save("./reconstruction_state.pkl")
    
    print("\n" + "="*60)
    print("STEP 6 COMPLETE!")
    print("="*60)
    print("✓ Reconstruction state initialized")
    print("✓ Camera poses stored for images", seed_pair)
    print("✓ 3D points and tracks stored")
    print("✓ Ready for incremental reconstruction")
    print("="*60)
    
    return state


def main():
    # Initialize from bootstrap
    state = initialize_reconstruction_state_from_bootstrap()
    
    # Demonstrate some queries
    print("\n[demo] Testing data structure queries...")
    
    # Get registered cameras
    registered = state.get_registered_cameras()
    print(f"\nRegistered cameras: {registered}")
    
    # Get points observed by first camera
    cam_idx = registered[0]
    point_indices, points_2d = state.get_points_observed_by_camera(cam_idx)
    print(f"\nCamera {cam_idx} observes {len(point_indices)} 3D points")
    
    # Get cameras observing first 3D point
    cameras, obs_2d = state.get_cameras_observing_point(0)
    print(f"\n3D point 0 is observed by cameras: {cameras}")
    
    # Get projection matrix
    P = state.get_projection_matrix(cam_idx)
    print(f"\nProjection matrix for camera {cam_idx}:")
    print(P)
    
    print("\n[demo] Data structure is ready for incremental SfM!")


if __name__ == "__main__":
    main()

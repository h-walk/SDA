#!/usr/bin/env python3
"""
Spectral Displacement (SD) Analysis Tool with Pattern Decomposition

This script performs SD analysis on molecular dynamics trajectories using the OVITO library.
It computes the SD, applies frequency filtering, performs pattern analysis, and outputs
visualization trajectories for the dominant spatial patterns.

Dependencies:
- numpy
- ovito
- matplotlib
- tqdm
- yaml
"""

import numpy as np
import os
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
from tqdm import tqdm
import yaml

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
warnings.filterwarnings('ignore', category=np.ComplexWarning)

from ovito.io import import_file

@dataclass
class Box:
    """Represents the simulation box with lengths, tilts, and the full cell matrix."""
    lengths: np.ndarray  # [lx, ly, lz]
    tilts: np.ndarray    # [xy, xz, yz]
    matrix: np.ndarray   # full 3x3 cell matrix

    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        matrix = cell.matrix.copy().astype(np.float32)
        lengths = np.array([
            matrix[0,0],
            matrix[1,1],
            matrix[2,2]
        ], dtype=np.float32)
        tilts = np.array([
            matrix[0,1],
            matrix[0,2],
            matrix[1,2]
        ], dtype=np.float32)
        return cls(lengths=lengths, tilts=tilts, matrix=matrix)
        
    def to_dict(self) -> Dict:
        return {
            'lengths': self.lengths,
            'tilts': self.tilts,
            'matrix': self.matrix
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Box':
        return cls(
            lengths=np.array(data['lengths'], dtype=np.float32),
            tilts=np.array(data['tilts'], dtype=np.float32),
            matrix=np.array(data['matrix'], dtype=np.float32)
        )

@dataclass
class Trajectory:
    """Stores trajectory data including positions, atom types, timesteps, and simulation box."""
    positions: np.ndarray  # Shape: (frames, atoms, 3)
    types: np.ndarray      # Shape: (atoms,)
    timesteps: np.ndarray  # Shape: (frames,)
    box: Box
    
    def __post_init__(self):
        if len(self.positions.shape) != 3:
            raise ValueError("Positions must be a 3D array (frames, atoms, xyz)")
        if len(self.types.shape) != 1:
            raise ValueError("Types must be a 1D array")
        if len(self.timesteps.shape) != 1:
            raise ValueError("Timesteps must be a 1D array")
    
    @property
    def n_frames(self) -> int:
        return len(self.timesteps)
    
    @property
    def n_atoms(self) -> int:
        return len(self.types)

class TrajectoryLoader:
    """Handles loading and saving of trajectory data."""
    def __init__(self, filename: str, dt: float = 1.0):
        self.filepath = Path(filename)
        self.dt = dt
    
    def _get_save_filenames(self) -> Dict[str, str]:
        prefix = self.filepath.stem
        return {
            'pos': f"{prefix}_pos.npy",
            'types': f"{prefix}_types.npy",
            'time': f"{prefix}_time.npy",
            'box': f"{prefix}_box.npz"
        }
    
    def _load_saved(self) -> Optional[Trajectory]:
        filenames = self._get_save_filenames()
        if all(Path(f).exists() for f in filenames.values()):
            logger.info("Loading previously saved processed data...")
            try:
                box_data = dict(np.load(filenames['box'], allow_pickle=True))
                return Trajectory(
                    positions=np.load(filenames['pos'], allow_pickle=True).astype(np.float32),
                    types=np.load(filenames['types'], allow_pickle=True),
                    timesteps=np.load(filenames['time'], allow_pickle=True).astype(np.float32),
                    box=Box.from_dict(box_data)
                )
            except Exception as e:
                logger.warning(f"Failed to load saved files: {e}")
                return None
        return None
    
    def _save_processed(self, traj: Trajectory) -> None:
        filenames = self._get_save_filenames()
        np.save(filenames['pos'], traj.positions)
        np.save(filenames['types'], traj.types)
        np.save(filenames['time'], traj.timesteps)
        np.savez(filenames['box'], **traj.box.to_dict())
        logger.info("Processed data saved for future use.")
    
    def load(self, reload: bool = False) -> Trajectory:
        if not reload:
            saved = self._load_saved()
            if saved is not None:
                return saved
        
        logger.info("Parsing trajectory from the original file...")
        pipeline = import_file(str(self.filepath))
        n_frames = pipeline.source.num_frames
        frame0 = pipeline.compute(0)
        n_atoms = len(frame0.particles.positions)
        
        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        types = frame0.particles.particle_types.array
        timesteps = np.arange(n_frames, dtype=np.float32) * self.dt
        box = Box.from_ovito(frame0.cell)
        
        for i in tqdm(range(n_frames), desc="Reading frames", ncols=80):
            frame = pipeline.compute(i)
            positions[i] = frame.particles.positions.array.astype(np.float32)
        
        traj = Trajectory(positions, types, timesteps, box)
        self._save_processed(traj)
        return traj

class SDCalculator:
    """Calculates the Spectral Displacement (SD) from trajectory data."""
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int):
        self.traj = traj
        self.system_size = np.array([nx, ny, nz], dtype=np.int32)
        
        # Compute reciprocal lattice vectors
        cell_mat = self.traj.box.matrix.astype(np.float32)
        self.a1 = cell_mat[:,0] / float(nx)
        self.a2 = cell_mat[:,1] / float(ny)
        self.a3 = cell_mat[:,2] / float(nz)
        
        volume = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / volume
        self.recip_vectors = np.vstack([b1, b2, b3]).astype(np.float32)
        
        logger.info("Reciprocal lattice vectors (2π/Å):\n{}".format(self.recip_vectors))
        logger.info("Reciprocal lattice vector magnitudes (2π/Å): {}".format(
            [f"{np.linalg.norm(b):.3f}" for b in self.recip_vectors]
        ))
    
    def get_k_path(self, direction: str, bz_coverage: float, n_k: int, 
                   lattice_parameter: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a path of k-points along a specified direction."""
        if isinstance(direction, str):
            direction = direction.lower()
            if direction == 'x':
                dir_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a1)
            elif direction == 'y':
                dir_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a2)
            elif direction == 'z':
                dir_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                computed_a = np.linalg.norm(self.a3)
            else:
                raise ValueError(f"Unknown direction '{direction}'")
        else:
            dir_vector = np.array(direction, dtype=np.float32)
            dir_vector /= np.linalg.norm(dir_vector)
            computed_a = np.linalg.norm(self.a1)

        if lattice_parameter is None:
            lattice_parameter = computed_a
            logger.info(f"Using computed lattice parameter: {lattice_parameter:.3f} Å")

        k_max = bz_coverage * (4 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)

        logger.info(f"Sampling k-path along '{direction}' with {n_k} points up to {k_max:.3f} (2π/Å)")
        return k_points, k_vectors

    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the Spectral Displacement (SD) for the provided k-points."""
        n_k = len(k_points)
        n_t = self.traj.n_frames
        n_atoms = self.traj.n_atoms

        # Calculate displacements from average positions
        disp = self.traj.positions - np.mean(self.traj.positions, axis=0)

        # Initialize arrays for global and per-atom SD
        sd = np.zeros((n_t, n_k, 3), dtype=np.complex64)
        sd_per_atom = np.zeros((n_t, n_k, n_atoms, 3), dtype=np.complex64)
        
        # Calculate timestep and frequencies
        dt_s = (self.traj.timesteps[1] - self.traj.timesteps[0]) * 1e-12
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12  # Convert to THz

        for i, k_vec in enumerate(tqdm(k_vectors, total=n_k, desc="Computing SD", ncols=80)):
            phases = np.exp(-1j * np.dot(self.traj.positions[0], k_vec))
            uk_t = disp * phases[np.newaxis, :, np.newaxis]
            uk_w = np.fft.fft(uk_t, axis=0)
            sd_per_atom[:, i, :, :] = uk_w
            sd[:, i, :] = uk_w.sum(axis=1) / np.sqrt(n_atoms)
        
        return sd, sd_per_atom, freqs

    def filter_sd(self, sd: np.ndarray, freqs: np.ndarray, freq_range: Tuple[float, float]) -> np.ndarray:
        """Applies Gaussian frequency filter to the SD."""
        filtered = sd.copy()
        f_min, f_max = freq_range
        f_center = (f_max + f_min) / 2
        f_sigma = (f_max - f_min) / 6
        
        freq_window = np.exp(-0.5 * ((freqs - f_center) / f_sigma)**2)
        freq_window = freq_window.reshape(-1, 1, *(1,) * (len(sd.shape) - 2))
        
        filtered *= freq_window
        logger.info(f"Applied Gaussian frequency filter: {f_center:.2f} THz ± {f_sigma:.2f} THz")
            
        return filtered

    def plot_sd(self, sd: np.ndarray, freqs: np.ndarray, k_points: np.ndarray,
               output: str, freq_range: Tuple[float, float], cmap: str = 'inferno',
               vmin: float = -6.5, vmax: float = 0) -> None:
        """Plots the Spectral Displacement."""
        if len(sd.shape) == 4:  # per-atom SD
            intensity = np.abs(np.sum(np.abs(sd)**2, axis=(2,3)))
        else:  # global SD
            intensity = np.abs(np.sum(sd * np.conj(sd), axis=-1))
            
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity /= max_intensity
            
        abs_freqs = np.abs(freqs)
        sorted_indices = np.argsort(abs_freqs)
        sorted_freqs = abs_freqs[sorted_indices]
        sorted_intensity = intensity[sorted_indices]
        
        log_intensity = np.log10(sorted_intensity + 1e-10)
        
        plt.figure(figsize=(10, 8))
        pcm = plt.pcolormesh(
            k_points, sorted_freqs, log_intensity,
            shading='gouraud',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        # Highlight filtered region
        f_min, f_max = freq_range
        rect = plt.Rectangle(
            (k_points[0], f_min),
            k_points[-1] - k_points[0],
            f_max - f_min,
            fill=False,
            edgecolor='white',
            linestyle='--',
            linewidth=2
        )
        plt.gca().add_patch(rect)
        
        plt.xlabel('k (2π/Å)')
        plt.ylabel('Frequency (THz)')
        plt.ylim(0, 40)
        plt.title('Spectral Displacement')
        plt.colorbar(pcm, label='log₁₀(Intensity)')
        
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SD plot saved as {output}")

class SingleFrequencyModeAnalyzer:
    """Analyzes N dominant spatial patterns at a single frequency."""
    
    def __init__(self, n_patterns: int = 3, tol: float = 1e-7, max_iter: int = 20):
        self.n_patterns = n_patterns
        self.tol = tol
        self.max_iter = max_iter
        
    def find_spatial_patterns(self, filtered_displacements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Finds N dominant spatial patterns in frequency-filtered displacements."""
        # Input validation
        if not isinstance(filtered_displacements, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if not np.any(np.isfinite(filtered_displacements)):
            raise ValueError("Input contains non-finite values")
            
        if filtered_displacements.ndim != 3:
            raise ValueError(f"Expected 3D array (frames, atoms, xyz), got shape {filtered_displacements.shape}")
            
        n_frames, n_atoms, dims = filtered_displacements.shape
        if dims != 3:
            raise ValueError(f"Expected 3 spatial dimensions, got {dims}")
            
        # Normalize input to improve numerical stability
        norm = np.linalg.norm(filtered_displacements)
        if norm < self.tol:
            raise ValueError("Input displacement amplitudes too small")
            
        filtered_displacements = filtered_displacements / norm
        
        flat_frames = filtered_displacements.reshape(n_frames, -1)
        n_space = n_atoms * dims
        
        patterns = np.zeros((self.n_patterns, n_space), dtype=np.float32)
        weights = np.zeros(self.n_patterns, dtype=np.float32)
        
        remaining_space = flat_frames.copy()
        
        for i in range(self.n_patterns):
            v = remaining_space[0].copy()
            v /= np.linalg.norm(v) + 1e-10
            
            for _ in range(self.max_iter):
                v_new = remaining_space.T @ (remaining_space @ v)
                norm = np.linalg.norm(v_new)
                if norm < self.tol:
                    break
                v_new /= norm
                
                if np.abs(np.sum(v * v_new)) > 1 - self.tol:
                    break
                v = v_new
            
            patterns[i] = v
            weights[i] = norm
            
            proj = remaining_space @ v.reshape(-1, 1)
            remaining_space -= proj @ v.reshape(1, -1)
        
        patterns = patterns.reshape(self.n_patterns, n_atoms, dims)
        weights /= weights.sum()
        
        return patterns, weights
            
    def create_pattern_trajectory(self, pattern: np.ndarray, ref_positions: np.ndarray, 
                                box: Box, atom_types: np.ndarray, amplitude: float = 1.0,
                                n_frames: int = 20) -> str:
        """Creates a LAMMPS trajectory showing the oscillation of a single pattern."""
        n_atoms = len(ref_positions)
        
        t = np.linspace(0, 2*np.pi, n_frames)
        factors = amplitude * np.sin(t)
        
        positions = ref_positions[np.newaxis, :, :] + \
                   pattern[np.newaxis, :, :] * factors[:, np.newaxis, np.newaxis]
        
        import time
        filename = f"pattern_{int(time.time())}.lammpstrj"
        
        with open(filename, 'w') as f:
            for frame in range(n_frames):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{frame}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{n_atoms}\n")
                f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                
                xy, xz, yz = box.tilts
                f.write(f"0.0 {box.lengths[0]} {xy}\n")
                f.write(f"0.0 {box.lengths[1]} {xz}\n")
                f.write(f"0.0 {box.lengths[2]} {yz}\n")
                
                f.write("ITEM: ATOMS id type x y z\n")
                for i, (pos, atype) in enumerate(zip(positions[frame], atom_types)):
                    f.write(f"{i+1} {atype} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
        
        return filename

def estimate_memory_usage(n_frames: int, n_atoms: int, n_k: int) -> float:
    """Estimates memory usage in GB."""
    bytes_per_float = 4  # Using float32
    bytes_per_complex = 8  # Complex64
    
    # Main arrays
    trajectory_size = n_frames * n_atoms * 3 * bytes_per_float
    sd_size = n_frames * n_k * 3 * bytes_per_complex
    sd_per_atom_size = n_frames * n_k * n_atoms * 3 * bytes_per_complex
    
    total_bytes = trajectory_size + sd_size + sd_per_atom_size
    return total_bytes / (1024**3)  # Convert to GB

def main():
    """Main function to execute the SD analysis workflow."""
    
    # Add memory estimation
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    except ImportError:
        available_memory = None
    parser = argparse.ArgumentParser(description='Spectral Displacement (SD) Analysis Tool')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file', default=None)
    parser.add_argument('--reload', action='store_true', help='Force reloading the trajectory data from the original file')
    args = parser.parse_args()

    # Load configuration parameters from YAML file if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config file '{args.config}': {e}")
    
    # Extract configuration parameters with default values
    dt = config.get('dt', 0.0075)
    nx = config.get('nx', 30)
    ny = config.get('ny', 30)
    nz = config.get('nz', 12)
    direction = config.get('direction', 'x')
    bz_coverage = config.get('bz_coverage', 0.5)
    n_kpoints = config.get('n_kpoints', 30)
    wmin = config.get('wmin', 35)
    wmax = config.get('wmax', 37)
    amplitude = config.get('amplitude', 0.05)
    n_patterns = config.get('n_patterns', 3)
    pattern_amplitude = config.get('pattern_amplitude', 1.0)
    lattice_parameter = config.get('lattice_parameter', None)

    logger.info("Initializing SD analysis...")
    loader = TrajectoryLoader(args.trajectory, dt=dt)
    traj = loader.load(reload=args.reload)

    logger.info(f"Trajectory loaded: {traj.n_frames} frames, {traj.n_atoms} atoms")
    logger.info(f"Simulation box lengths (Å): {traj.box.lengths}")
    logger.info(f"Simulation box tilts: {traj.box.tilts}")

    calculator = SDCalculator(traj, nx, ny, nz)

    # Check memory requirements
    estimated_memory = estimate_memory_usage(traj.n_frames, traj.n_atoms, n_kpoints)
    logger.info(f"Estimated memory requirement: {estimated_memory:.1f} GB")
    
    if available_memory is not None:
        logger.info(f"Available system memory: {available_memory:.1f} GB")
        if estimated_memory > available_memory * 0.9:  # Leave 10% buffer
            logger.warning("Warning: Memory requirements may exceed available memory!")
    
    k_points, k_vectors = calculator.get_k_path(
        direction=direction,
        bz_coverage=bz_coverage,
        n_k=n_kpoints,
        lattice_parameter=lattice_parameter
    )

    sd, sd_per_atom, freqs = calculator.calculate_sd(k_points, k_vectors)
    
    # Plot unfiltered SD
    calculator.plot_sd(
        sd, freqs, k_points,
        output='sd_global.png',
        freq_range=(wmin, wmax),
        vmin=-6.5,
        vmax=0
    )

    # Apply frequency filter
    freq_range = (wmin, wmax)
    filtered_sd = calculator.filter_sd(sd, freqs, freq_range)
    filtered_sd_per_atom = calculator.filter_sd(sd_per_atom, freqs, freq_range)

    # Plot filtered SD
    calculator.plot_sd(
        filtered_sd, freqs, k_points,
        output='sd_filtered.png',
        freq_range=freq_range,
        vmin=-6.5,
        vmax=0
    )

    # Inverse FFT to get time domain
    try:
        filtered_time = np.fft.ifft(filtered_sd_per_atom, axis=0)
        if not np.any(np.isfinite(filtered_time)):
            raise ValueError("Non-finite values in FFT result")
            
        # Take real part and check for numerical stability
        filtered_time = np.real(filtered_time).astype(np.float32)
        
        # Reshape and handle k-points by averaging
        n_frames, n_k, n_atoms, dims = filtered_time.shape
        filtered_time = np.mean(filtered_time, axis=1)  # Average over k-points
        
        # Check for valid data
        if not np.any(np.isfinite(filtered_time)):
            raise ValueError("Non-finite values after k-point averaging")
            
        logger.info(f"Filtered data shape: {filtered_time.shape}")
        
    except Exception as e:
        logger.error(f"Error in FFT processing: {e}")
        raise

    # Pattern analysis
    analyzer = SingleFrequencyModeAnalyzer(n_patterns=n_patterns)
    patterns, weights = analyzer.find_spatial_patterns(filtered_time)
    
    # Print pattern weights
    print("\nPattern Analysis Results:")
    for i, weight in enumerate(weights):
        print(f"Pattern {i+1}: {weight*100:.1f}% contribution")
    
    # Create visualization trajectories for each pattern
    ref_positions = np.mean(traj.positions, axis=0)
    for i, pattern in enumerate(patterns):
        filename = analyzer.create_pattern_trajectory(
            pattern,
            ref_positions=ref_positions,
            box=traj.box,
            atom_types=traj.types,
            amplitude=pattern_amplitude
        )
        print(f"Created visualization trajectory for pattern {i+1}: {filename}")

    logger.info("SD analysis and pattern decomposition completed successfully.")

if __name__ == "__main__":
    main()

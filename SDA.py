"""
Spectral Displacement (SD) Analysis Tool with Time-Domain Frequency Filtering
Optional 3D dispersion plotting + saving an NPZ archive of kx, ky, freq, and amplitude.

Only writes out the trajectory npy files once to avoid clutter.
"""

import numpy as np
from pathlib import Path
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
import logging
import argparse
import yaml
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
except ImportError as e:
    logging.error(f"Failed to import matplotlib or mpl_toolkits: {e}")
    logging.error("Please install it using: pip install matplotlib")
    raise

try:
    import ovito
    from ovito.io import import_file
    from ovito.modifiers import UnwrapTrajectoriesModifier
except ImportError as e:
    logging.error(f"Failed to import OVITO: {e}")
    logging.error("Please install it using: pip install ovito") 
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific OVITO warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')


@dataclass
class Box:
    lengths: np.ndarray
    tilts: np.ndarray
    matrix: np.ndarray

    @classmethod
    def from_ovito(cls, cell) -> 'Box':
        matrix = cell.matrix.copy().astype(np.float32)
        lengths = np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]], dtype=np.float32)
        tilts = np.array([matrix[0, 1], matrix[0, 2], matrix[1, 2]], dtype=np.float32)
        return cls(lengths=lengths, tilts=tilts, matrix=matrix)


@dataclass
class Trajectory:
    positions: np.ndarray      # (n_frames, n_atoms, 3)
    velocities: np.ndarray     # (n_frames, n_atoms, 3)
    types: np.ndarray          # (n_atoms,)
    timesteps: np.ndarray      # (n_frames,)
    box: Box

    def __post_init__(self):
        if self.positions.ndim != 3:
            raise ValueError("Positions must be a 3D array (frames, atoms, xyz)")
        if self.velocities.ndim != 3:
            raise ValueError("Velocities must be a 3D array (frames, atoms, xyz)")
        if self.types.ndim != 1:
            raise ValueError("Types must be a 1D array")
        if self.timesteps.ndim != 1:
            raise ValueError("Timesteps must be a 1D array")

    @property
    def n_frames(self) -> int:
        return len(self.timesteps)

    @property
    def n_atoms(self) -> int:
        return len(self.types)


class TrajectoryLoader:
    def __init__(self, filename: str, dt: float = 1.0, file_format: str = 'auto'):
        """
        Args:
            filename: Path to the trajectory file.
            dt: Time step in picoseconds.
            file_format: Format of the input file ('auto', 'lammps', 'vasp_outcar')
        """
        if dt <= 0:
            raise ValueError("Time step (dt) must be positive.")
        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        self.dt = dt
        
        # Validate and set file format
        valid_formats = ['auto', 'lammps', 'vasp_outcar']
        if file_format not in valid_formats:
            raise ValueError(f"Unsupported file format. Must be one of: {valid_formats}")
        self.file_format = file_format

    def _detect_file_format(self) -> str:
        """Automatically detect if file is VASP OUTCAR or LAMMPS trajectory."""
        if self.file_format != 'auto':
            return self.file_format

        # Check file extension first
        if self.filepath.suffix.lower() == '.outcar':
            return 'vasp_outcar'
        
        # Peek at content for OUTCAR markers
        try:
            with open(self.filepath, 'r') as f:
                first_line = f.readline().strip()
                if 'OUTCAR' in first_line or 'vasp' in first_line.lower():
                    return 'vasp_outcar'
        except:
            pass
        
        return 'lammps'

    def load(self) -> Trajectory:
        """
        Checks if precomputed .npy files exist. If they do, loads from them.
        Otherwise, loads and unwraps the trajectory via OVITO.
        """
        base_path = self.filepath.with_suffix('')
        npy_files = {
            'positions': base_path.with_suffix('.positions.npy'),
            'velocities': base_path.with_suffix('.velocities.npy'),
            'types': base_path.with_suffix('.types.npy'),
        }

        if all(f.exists() for f in npy_files.values()):
            logger.info("Found existing .npy files. Attempting to load them.")
            try:
                positions = np.load(npy_files['positions'])
                velocities = np.load(npy_files['velocities'])
                types = np.load(npy_files['types'])

                # We still need the box from the first frame
                pipeline = import_file(str(self.filepath))
                pipeline.modifiers.append(UnwrapTrajectoriesModifier())
                frame0 = pipeline.compute(0)
                box = Box.from_ovito(frame0.cell)

                n_frames = positions.shape[0]
                timesteps = np.arange(n_frames, dtype=np.float32)
                logger.info("Loaded trajectory from .npy files.")
                return Trajectory(positions, velocities, types, timesteps, box)

            except Exception as e:
                logger.warning(f"Failed to load from .npy files: {e}")
                logger.info("Falling back to OVITO-based load.")
                return self._load_via_ovito()
        else:
            logger.info("No .npy trajectory files found. Loading via OVITO.")
            return self._load_via_ovito()

    def _load_via_ovito(self) -> Trajectory:
        """Loads and unwraps the trajectory via OVITO."""
        logger.info("Loading and unwrapping trajectory with OVITO.")
        format_type = self._detect_file_format()
        logger.info(f"Detected file format: {format_type}")
        
        # Import with columns explicitly specified for OUTCAR to ensure velocities
        if format_type == 'vasp_outcar':
            pipeline = import_file(str(self.filepath), 
                                   columns=["Particle Type", "Position.X", "Position.Y", "Position.Z", 
                                            "Velocity.X", "Velocity.Y", "Velocity.Z"])
        else:
            pipeline = import_file(str(self.filepath))
            
        pipeline.modifiers.append(UnwrapTrajectoriesModifier())

        n_frames = pipeline.source.num_frames
        frame0 = pipeline.compute(0)
        n_atoms = len(frame0.particles.positions)

        if not hasattr(frame0.particles, 'velocities'):
            raise ValueError("No velocity data found in the trajectory file. "
                             "For VASP files, ensure you're using an OUTCAR file with velocity data.")

        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

        for i in tqdm(range(n_frames), desc="Loading frames", unit="frame"):
            frame = pipeline.compute(i)
            positions[i] = frame.particles.positions.array.astype(np.float32)
            velocities[i] = frame.particles.velocities.array.astype(np.float32)

        types = frame0.particles.particle_types.array
        timesteps = np.arange(n_frames, dtype=np.float32)
        box = Box.from_ovito(frame0.cell)

        logger.info("Successfully loaded trajectory with OVITO.")
        return Trajectory(positions, velocities, types, timesteps, box)

    def save_trajectory_npy(self, traj: Trajectory) -> None:
        """
        Saves the trajectory data to .npy files only if not already present.
        """
        base_path = self.filepath.parent / self.filepath.stem
        traj_files_exist = all(
            (base_path.with_suffix(suffix)).exists() 
            for suffix in ['.positions.npy', '.velocities.npy', '.types.npy']
        )
        if traj_files_exist:
            logger.info("Trajectory npy files already exist; skipping.")
            return

        logger.info("Saving trajectory data to .npy files (positions, velocities, types).")
        np.save(base_path.with_suffix('.positions.npy'), traj.positions)
        np.save(base_path.with_suffix('.velocities.npy'), traj.velocities)
        np.save(base_path.with_suffix('.types.npy'), traj.types)

        mean_positions = np.mean(traj.positions, axis=0)
        displacements = traj.positions - mean_positions[None, :, :]
        np.save(base_path.with_suffix('.mean_positions.npy'), mean_positions)
        np.save(base_path.with_suffix('.displacements.npy'), displacements)
        logger.info("Trajectory data saved to .npy files.")

    # Note: The function save_sd_arrays is now deprecated per new requirements.
    # We no longer save individual SD .npy files for each direction.
    # def save_sd_arrays(self, ...): 
    #     ...


def parse_direction(direction: Union[str, int, float, List[float], Dict[str, float]]) -> np.ndarray:
    """
    Parse a direction specification.
    If a numeric value (int or float) is provided, it is interpreted as an angle in degrees
    measured from the positive x-axis (unit circle convention).
    If a string can be converted to a float, it is also interpreted as an angle in degrees.
    Otherwise, falls back to legacy behavior.
    """
    if isinstance(direction, (int, float)):
        # Angle in degrees.
        rad = np.deg2rad(direction)
        vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
    elif isinstance(direction, str):
        try:
            angle = float(direction)
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        except ValueError:
            # Fallback to legacy string mapping.
            direction_map = {'x': [1.0, 0.0, 0.0],
                             'y': [0.0, 1.0, 0.0],
                             'z': [0.0, 0.0, 1.0]}
            if direction.lower() not in direction_map:
                raise ValueError(f"Unknown direction string: {direction}")
            vec = np.array(direction_map[direction.lower()], dtype=np.float32)
    elif isinstance(direction, (list, tuple, np.ndarray)):
        if len(direction) == 1:
            # Single value in list: assume angle in degrees.
            angle = float(direction[0])
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        elif len(direction) == 3:
            vec = np.array(direction, dtype=np.float32)
        else:
            raise ValueError("Direction vector must have 1 or 3 components")
    elif isinstance(direction, dict):
        if 'angle' in direction:
            angle = float(direction['angle'])
            rad = np.deg2rad(angle)
            vec = np.array([np.cos(rad), np.sin(rad), 0.0], dtype=np.float32)
        else:
            vec = np.array([direction.get('h', 0.0),
                            direction.get('k', 0.0),
                            direction.get('l', 0.0)], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported direction format: {type(direction)}")

    if np.linalg.norm(vec) < 1e-10:
        raise ValueError("Direction vector cannot be zero")
    return vec


class SDCalculator:
    def __init__(self,
                 traj: Trajectory,
                 nx: int,
                 ny: int,
                 nz: int,
                 dt_ps: float,
                 use_velocities: bool = False):
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("System size dimensions must be positive integers")
        self.traj = traj
        self.use_velocities = use_velocities
        self.dt_ps = dt_ps

        cell_mat = self.traj.box.matrix.astype(np.float32)
        self.a1 = cell_mat[:, 0] / float(nx)
        self.a2 = cell_mat[:, 1] / float(ny)
        self.a3 = cell_mat[:, 2] / float(nz)

        volume = np.dot(self.a1, np.cross(self.a2, self.a3))
        b1 = 2 * np.pi * np.cross(self.a2, self.a3) / volume
        b2 = 2 * np.pi * np.cross(self.a3, self.a1) / volume
        b3 = 2 * np.pi * np.cross(self.a1, self.a2) / volume
        self.recip_vectors = np.vstack([b1, b2, b3]).astype(np.float32)

    def get_k_path(self,
                   direction: Union[str, int, float, List[float], np.ndarray],
                   bz_coverage: float,
                   n_k: int,
                   lattice_parameter: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        dir_vector = parse_direction(direction)
        norm_v = np.linalg.norm(dir_vector)
        dir_vector /= norm_v

        if lattice_parameter is None:
            lattice_parameter = np.linalg.norm(self.a1)
            logger.info(f"Using lattice parameter: {lattice_parameter:.3f} Å")

        k_max = bz_coverage * (2 * np.pi / lattice_parameter)
        k_points = np.linspace(0, k_max, n_k, dtype=np.float32)
        k_vectors = np.outer(k_points, dir_vector).astype(np.float32)
        return k_points, k_vectors

    def calculate_sd(self, k_points: np.ndarray, k_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_t = self.traj.n_frames
        mean_positions = np.mean(self.traj.positions, axis=0)
        sed = np.zeros((n_t, len(k_points), 3), dtype=np.complex64)

        if self.use_velocities:
            logger.info("Using velocities for SD calculation.")
            data_array = self.traj.velocities
        else:
            logger.info("Using displacements for SD calculation.")
            data_array = self.traj.positions - mean_positions

        for ik, kvec in enumerate(tqdm(k_vectors, desc="Processing k-points", unit="k-point")):
            phase = np.exp(1j * np.dot(mean_positions, kvec))
            for alpha in range(3):
                sed[:, ik, alpha] = np.sum(data_array[..., alpha] * phase, axis=1)

        dt_s = self.dt_ps * 1e-12
        sed_w = np.fft.fft(sed, axis=0)
        freqs = np.fft.fftfreq(n_t, d=dt_s) * 1e-12
        return sed_w, freqs

    def plot_sed(self,
                 sed: np.ndarray,
                 freqs: np.ndarray,
                 k_points: np.ndarray,
                 output: str,
                 direction_label: str = '',
                 cmap: str = 'inferno',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 global_max_intensity: Optional[float] = None,
                 highlight_region: Optional[Dict[str, Tuple[float, float]]] = None,
                 max_freq: Optional[float] = None) -> None:
        try:
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            sed = sed[pos_mask]

            intensity = np.abs(sed).sum(axis=-1).real.astype(np.float32)
            k_mesh, f_mesh = np.meshgrid(k_points, freqs)

            if global_max_intensity is not None and global_max_intensity > 0:
                intensity /= global_max_intensity
            else:
                max_int = np.max(intensity)
                if max_int > 0:
                    intensity /= max_int

            sqrt_intensity = np.sqrt(intensity + 1e-20)
            if vmin is None:
                vmin = np.percentile(sqrt_intensity[sqrt_intensity > 0], 1)
            if vmax is None:
                vmax = np.percentile(sqrt_intensity[sqrt_intensity > 0], 99)

            plt.figure(figsize=(10, 8))
            pcm = plt.pcolormesh(k_mesh, f_mesh, sqrt_intensity,
                                 shading='gouraud', cmap=cmap,
                                 vmin=vmin, vmax=vmax)

            if highlight_region:
                fr = highlight_region.get('freq_range')
                kr = highlight_region.get('k_range')
                if fr and kr:
                    f_min, f_max = fr
                    k_min, k_max = kr
                    rect = plt.Rectangle((k_min, f_min), k_max - k_min, f_max - f_min,
                                         fill=False, edgecolor='white', linestyle='--', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(k_max + 0.05, 0.5 * (f_max + f_min),
                             (f'Selected\nRegion\n{f_min}-{f_max} THz\n{k_min}-{k_max} (2π/Å)'),
                             color='white', va='center', fontsize=8)

            plt.xlabel('k (2π/Å)')
            plt.ylabel('Frequency (THz)')
            if max_freq is not None and max_freq > 0:
                plt.ylim(0, max_freq)
            else:
                plt.ylim(0, np.max(freqs))

            plt.title(f'Spectral Energy Density {direction_label}')
            plt.colorbar(pcm, label='√Intensity (arb. units)')
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SED plot saved as {output}")
        except Exception as e:
            logger.error(f"Failed to create SED plot: {e}")
            raise


class TimeDomainFilter:
    def __init__(self, dt_ps: float):
        if dt_ps <= 0:
            raise ValueError("Time step must be positive.")
        self.dt_ps = dt_ps
        self.dt_s = dt_ps * 1e-12

    def filter_in_frequency(self,
                            data: np.ndarray,
                            w_min: float,
                            w_max: float) -> np.ndarray:
        n_frames, n_atoms, _ = data.shape
        freqs = np.fft.fftfreq(n_frames, d=self.dt_s) * 1e-12

        w_center = 0.5 * (w_min + w_max)
        w_sigma = (w_max - w_min) / 6.0 if w_max > w_min else 0.0
        logger.info(f"Applying Gaussian filter: center={w_center:.2f} THz, sigma={w_sigma:.2f} THz")

        if w_sigma < 1e-14:
            logger.warning("Filter width too small; returning original data.")
            return data.copy()

        freq_window = np.exp(-0.5 * ((freqs - w_center) / w_sigma) ** 2)
        filtered_data = np.zeros_like(data, dtype=np.float32)

        for i_atom in tqdm(range(n_atoms), desc="Filtering atoms", unit="atom"):
            for alpha in range(3):
                time_series = data[:, i_atom, alpha]
                fft_vals = np.fft.fft(time_series)
                fft_filtered = fft_vals * freq_window
                filtered_ts = np.fft.ifft(fft_filtered)
                filtered_data[:, i_atom, alpha] = filtered_ts.real

        return filtered_data


def write_filtered_trajectory(filename: str,
                              ref_positions: np.ndarray,
                              box: Box,
                              filtered_data: np.ndarray,
                              types: np.ndarray,
                              dt_ps: float,
                              start_time_ps: float = 0.0):
    n_frames, n_atoms, _ = filtered_data.shape
    xy, xz, yz = box.tilts
    xlo = 0.0
    xhi = box.lengths[0]
    ylo = 0.0
    yhi = box.lengths[1]
    zlo = 0.0
    zhi = box.lengths[2]

    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)

    logger.info(f"Writing filtered trajectory to {filename}")
    with open(filename, 'w') as f:
        for frame_idx in tqdm(range(n_frames), desc="Writing frames", unit="frame"):
            current_time_ps = start_time_ps + frame_idx * dt_ps
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{current_time_ps:.6f}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_atoms}\n")
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write(f"{xlo_bound:.6f} {xhi_bound:.6f} {xy:.6f}\n")
            f.write(f"{ylo_bound:.6f} {yhi_bound:.6f} {xz:.6f}\n")
            f.write(f"{zlo:.6f} {zhi:.6f} {yz:.6f}\n")

            coords = ref_positions + filtered_data[frame_idx]
            f.write("ITEM: ATOMS id type x y z\n")
            for i_atom in range(n_atoms):
                x, y, z = coords[i_atom]
                atype = types[i_atom]
                f.write(f"{i_atom+1} {atype} {x:.6f} {y:.6f} {z:.6f}\n")


def gather_3d_data(k_vectors_list: List[np.ndarray],
                   freqs_list: List[np.ndarray],
                   sed_list: List[np.ndarray],
                   intensity_threshold: float = 0.01):
    """
    Aggregates data into 4 arrays: kx_vals, ky_vals, freq_vals, amp_vals,
    by summing amplitude over the 3 polarization directions (i.e., axis=-1).
    """
    kx_vals = []
    ky_vals = []
    freq_vals = []
    amp_vals = []

    for freqs, kvecs, sed in zip(freqs_list, k_vectors_list, sed_list):
        # sed is shape (n_freq, n_k, 3). Sum over alpha for total amplitude:
        intensity_3d = np.abs(sed).sum(axis=-1)  # shape (n_freq, n_k)
        n_freq = len(freqs)
        n_k = len(kvecs)

        for i_f in range(n_freq):
            for i_k in range(n_k):
                amp = intensity_3d[i_f, i_k]
                kx_vals.append(kvecs[i_k][0])
                ky_vals.append(kvecs[i_k][1])
                freq_vals.append(freqs[i_f])
                amp_vals.append(amp)

    return (np.array(kx_vals, dtype=np.float32),
            np.array(ky_vals, dtype=np.float32),
            np.array(freq_vals, dtype=np.float32),
            np.array(amp_vals, dtype=np.float32))


def plot_3d_dispersion(kx_vals: np.ndarray,
                       ky_vals: np.ndarray,
                       freq_vals: np.ndarray,
                       amp_vals: np.ndarray,
                       output_path: str):
    """
    Creates a 3D scatter plot of (kx, ky, freq) colored by the log amplitude.
    """
    if len(kx_vals) == 0:
        logger.warning("No 3D data points to plot. Check intensity_threshold or directions.")
        return

    amp_vals[amp_vals < 1e-20] = 1e-20

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter, colored by log amplitude
    sc = ax.scatter(
        kx_vals, ky_vals, freq_vals,
        c=np.log(amp_vals),
        cmap='plasma',
        alpha=0.7,
        marker='o',
        s=10,
        edgecolors='none',
        norm=mcolors.Normalize(vmin=np.log(amp_vals.min()),
                               vmax=np.log(amp_vals.max()))
    )

    ax.set_xlabel(r'$k_x$ (2$\pi$/Å)')
    ax.set_ylabel(r'$k_y$ (2$\pi$/Å)')
    ax.set_zlabel('Frequency (THz)')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('log(Amplitude)')

    plt.title('3D Dispersion Visualization')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"3D dispersion plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Spectral Displacement (SD) Analysis Tool with Time-Domain Filter')
    parser.add_argument('trajectory', help='Path to the trajectory file')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default config: note that 'directions' now expects a list of angles in degrees.
    config = {
        'dt': 0.005,
        'nx': 60,
        'ny': 60,
        'nz': 1,
        'directions': None,  # e.g., [0, 45, 90]
        'n_kpoints': 60,
        'bz_coverage': 1.0,
        'max_freq': 50,
        'wmin': 0,
        'wmax': 50,
        'kmin': None,
        'kmax': None,
        'amplitude': 1.0,
        'lattice_parameter': None,
        'do_filtering': False,
        'do_reconstruction': False,
        'use_velocities': True,
        'save_npy': True,       # For trajectory only; SD arrays per direction will no longer be saved.
        '3D_Dispersion': False
    }

    if args.config:
        try:
            with open(args.config, 'r') as fh:
                user_config = yaml.safe_load(fh) or {}
                config.update(user_config)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Use provided directions (angles in degrees) or default to 0 degrees.
    directions = config['directions'] if config.get('directions') is not None else [0]

    try:
        # Load trajectory
        loader = TrajectoryLoader(args.trajectory, dt=config['dt'])
        traj = loader.load()

        # Save trajectory npy files just once (if save_npy is True)
        if config['save_npy']:
            loader.save_trajectory_npy(traj)

        sd_calc = SDCalculator(
            traj=traj,
            nx=config['nx'],
            ny=config['ny'],
            nz=config['nz'],
            dt_ps=config['dt'],
            use_velocities=config['use_velocities']
        )

        # Collect data for final 3D dispersion plot (if requested)
        all_kvecs = []
        all_freqs = []
        all_seds = []

        for i_dir, angle in enumerate(directions, start=1):
            # Here, angle is interpreted as degrees from the x-axis.
            dir_vec = parse_direction(angle)
            angle_str = f"{float(angle):.1f}"
            logger.info(f"Processing SD for angle: {angle_str}°")

            k_points, k_vectors = sd_calc.get_k_path(
                direction=angle,  # passing the angle directly
                bz_coverage=config['bz_coverage'],
                n_k=config['n_kpoints'],
                lattice_parameter=config['lattice_parameter']
            )

            sd, freqs = sd_calc.calculate_sd(k_points, k_vectors)
            max_intensity = np.max(np.abs(sd.sum(axis=-1)))

            all_kvecs.append(k_vectors)  # shape (n_k, 3)
            all_freqs.append(freqs)      # shape (n_freq,)
            all_seds.append(sd)          # shape (n_freq, n_k, 3)

            # Plot and save SED for this angle using the angle in the filename.
            data_type = 'vel' if config['use_velocities'] else 'disp'
            sed_plot_path = out_dir / f"{i_dir:03d}_sd_global_{data_type}_{angle_str}deg.png"
            sd_calc.plot_sed(
                sed=sd,
                freqs=freqs,
                k_points=k_points,
                output=str(sed_plot_path),
                direction_label=f"(angle: {angle_str}°)",
                global_max_intensity=max_intensity,
                max_freq=config['max_freq']
            )

            # Do NOT save individual SD .npy files per angle per new requirements.

        if config.get('3D_Dispersion', False):
            logger.info("Generating 3D dispersion data and plot.")
            kx_vals, ky_vals, freq_vals, amp_vals = gather_3d_data(
                k_vectors_list=all_kvecs,
                freqs_list=all_freqs,
                sed_list=all_seds,
                intensity_threshold=0.01
            )

            data_3d_path = out_dir / "3d_dispersion_data.npz"
            np.savez(
                data_3d_path,
                kx=kx_vals,
                ky=ky_vals,
                freq=freq_vals,
                amp=amp_vals
            )
            logger.info(f"Saved 3D dispersion data to {data_3d_path}")

            three_d_plot_path = out_dir / '3d_dispersion.png'
            plot_3d_dispersion(
                kx_vals=kx_vals,
                ky_vals=ky_vals,
                freq_vals=freq_vals,
                amp_vals=amp_vals,
                output_path=str(three_d_plot_path)
            )

        if config['do_filtering']:
            logger.info("Applying time-domain frequency filter.")
            filter_obj = TimeDomainFilter(dt_ps=config['dt'])
            if config['use_velocities']:
                data_array = traj.velocities
                logger.info("Filtering velocities in time-domain.")
            else:
                mean_positions = np.mean(traj.positions, axis=0)
                data_array = traj.positions - mean_positions
                logger.info("Filtering displacements in time-domain.")

            filtered_data = filter_obj.filter_in_frequency(
                data=data_array,
                w_min=config['wmin'],
                w_max=config['wmax']
            )

            if config['amplitude'] != 1.0:
                current_rms = np.sqrt(np.mean(filtered_data**2))
                if current_rms > 1e-14:
                    factor = config['amplitude'] / current_rms
                    filtered_data *= factor
                    logger.info(f"Scaled filtered data by factor {factor:.3f}")

            if config['do_reconstruction']:
                logger.info("Writing filtered trajectory to LAMMPS.")
                if config['use_velocities']:
                    ref_pos = traj.positions[0]
                else:
                    ref_pos = np.mean(traj.positions, axis=0)
                out_filtered = out_dir / 'filtered_time_domain.lammpstrj'
                write_filtered_trajectory(
                    filename=str(out_filtered),
                    ref_positions=ref_pos,
                    box=traj.box,
                    filtered_data=filtered_data,
                    types=traj.types,
                    dt_ps=config['dt'],
                    start_time_ps=0.0
                )

        logger.info("SD analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

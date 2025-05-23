"""
Core SED calculation engine.
"""
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import logging
from pathlib import Path
from tqdm import tqdm

from .trajectory import Trajectory
from .sed import SED
from ..utils.helpers import parse_direction
from ..io.writer import out_to_qdump
from ..visualization import SEDPlotter

logger = logging.getLogger(__name__)

class SEDCalculator:
    def __init__(self, traj: Trajectory, nx: int, ny: int, nz: int, 
                 use_displacements: bool = False, dt_ps: Optional[float] = None):
        if not (nx > 0 and ny > 0 and nz > 0):
            raise ValueError("System dimensions (nx, ny, nz) must be positive.")
        self.traj = traj
        self.use_displacements = use_displacements
        
        if dt_ps is not None:
            logger.warning("Explicitly providing dt_ps to SEDCalculator is deprecated. "
                           "The timestep will be taken from the Trajectory object. "
                           "The provided dt_ps will override the Trajectory's dt_ps.")
            self.dt_ps = dt_ps
        elif hasattr(self.traj, 'dt_ps') and self.traj.dt_ps is not None:
            self.dt_ps = self.traj.dt_ps
        else:
            # This case should ideally not be reached if TrajectoryLoader always sets dt_ps
            raise ValueError("Timestep dt_ps not found in Trajectory object and not provided to SEDCalculator.")

        if self.dt_ps <= 0:
            raise ValueError("Timestep dt_ps must be positive.")

        L1, L2, L3 = self.traj.box_matrix[0,:], self.traj.box_matrix[1,:], self.traj.box_matrix[2,:]
        self.a1, self.a2, self.a3 = L1/nx, L2/ny, L3/nz
        
        if any(np.linalg.norm(v) < 1e-9 for v in [self.a1, self.a2, self.a3]):
            raise ValueError("One or more primitive vectors (a1,a2,a3) near zero. Check nx,ny,nz or box matrix.")

        vol_prim = np.abs(np.dot(self.a1, np.cross(self.a2, self.a3)))
        if np.isclose(vol_prim, 0): 
            mat_A = np.vstack([self.a1, self.a2, self.a3])
            if np.linalg.matrix_rank(mat_A) < 3 or np.isclose(np.linalg.det(mat_A),0):
                 raise ValueError(f"Primitive cell vectors coplanar/collinear; volume zero ({vol_prim:.2e}).")
            else: logger.warning(f"Primitive cell volume very small ({vol_prim:.2e}).")

        self.b1 = (2*np.pi/vol_prim) * np.cross(self.a2, self.a3)
        self.b2 = (2*np.pi/vol_prim) * np.cross(self.a3, self.a1)
        self.b3 = (2*np.pi/vol_prim) * np.cross(self.a1, self.a2)
        self.recip_vecs_prim = np.vstack([self.b1, self.b2, self.b3]).astype(np.float32)

    def _calculate_sed_for_group(self, k_vectors_3d: np.ndarray, 
                                   group_atom_indices: np.ndarray, 
                                   mean_pos_all: np.ndarray) -> np.ndarray: # Returns complex SED for the group
        """Helper to calculate complex SED for a specific group of atoms."""
        n_t = self.traj.n_frames
        
        if group_atom_indices.size == 0:
            return np.zeros((n_t, len(k_vectors_3d), 3), dtype=np.complex64)

        mean_pos_group = mean_pos_all[group_atom_indices]
        
        if self.use_displacements:
            data_ft_group = (self.traj.positions[:, group_atom_indices, :] - mean_pos_group[None, :, :])
        else:
            data_ft_group = self.traj.velocities[:, group_atom_indices, :]

        n_k_vecs = len(k_vectors_3d)
        sed_tk_pol_group = np.zeros((n_t, n_k_vecs, 3), dtype=np.complex64)

        # Optimized phase factor calculation and summation using einsum
        phase_factors_exp = np.exp(1j * np.dot(k_vectors_3d, mean_pos_group.T))

        for pol_axis in range(3):
            sed_tk_pol_group[:, :, pol_axis] = np.einsum('ta,ak->tk', data_ft_group[:, :, pol_axis], phase_factors_exp.T, optimize=True)

        sed_wk_group = np.fft.fft(sed_tk_pol_group, axis=0) / n_t if n_t > 0 else np.array([],dtype=np.complex64).reshape(0,n_k_vecs,3)
        return sed_wk_group.astype(np.complex64)

    def get_k_path(self, direction_spec: Union[str, int, float, List[float], Dict[str, float], np.ndarray],
                   bz_coverage: float, n_k: int, lat_param: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]: # Returns (k_magnitudes, k_vectors_3d)
        k_dir_unit = parse_direction(direction_spec)

        if lat_param is None or lat_param <= 1e-6:
            # Calculate the projection of the k-direction onto the reciprocal lattice vectors
            # This gives the true reciprocal lattice extent in the specified direction
            b_proj_x = np.dot(k_dir_unit, self.b1)
            b_proj_y = np.dot(k_dir_unit, self.b2) 
            b_proj_z = np.dot(k_dir_unit, self.b3)
            
            # Find the largest projection magnitude to determine the BZ boundary
            projections = [abs(b_proj_x), abs(b_proj_y), abs(b_proj_z)]
            max_projection = max(projections)
            
            if max_projection > 1e-6:
                # Use the reciprocal projection as the characteristic k-extent
                recip_extent = max_projection
                logger.info(f"Using directional reciprocal lattice projection ({recip_extent:.3f} 2π/Å) for k-path.")
                logger.info(f"  Direction projections: b1·k̂={b_proj_x:.3f}, b2·k̂={b_proj_y:.3f}, b3·k̂={b_proj_z:.3f}")
            else:
                # Fallback to |a1| if reciprocal projections are too small
                norm_a1 = np.linalg.norm(self.a1)
                if norm_a1 > 1e-6: 
                    recip_extent = 2*np.pi / norm_a1
                    logger.warning(f"Reciprocal projections too small, using |a1| fallback ({norm_a1:.3f} Å → {recip_extent:.3f} 2π/Å).")
                else: 
                    raise ValueError("Invalid/small lattice_param for k-path & reciprocal projections too small for auto-detection.")
        else:
            # Use provided lattice parameter (convert to reciprocal space)
            recip_extent = 2*np.pi / lat_param
            logger.info(f"Using provided lattice parameter ({lat_param:.3f} Å → {recip_extent:.3f} 2π/Å) for k-path.")
        
        k_max_val = bz_coverage * recip_extent
        if n_k < 1: 
            raise ValueError("n_k (k-points) must be >= 1.")
        k_mags = np.linspace(0, k_max_val, n_k, dtype=np.float32) if n_k > 1 else np.array([0.0 if np.isclose(k_max_val,0) else k_max_val], dtype=np.float32)
        k_vecs = np.outer(k_mags, k_dir_unit).astype(np.float32)
        return k_mags, k_vecs

    def get_k_grid(self, 
                   plane: str, 
                   k_range_x: Tuple[float, float], 
                   k_range_y: Tuple[float, float], 
                   n_kx: int, 
                   n_ky: int, 
                   k_fixed_val: float = 0.0
                   ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Generates a 2D grid of k-points in a specified plane.

        Args:
            plane: The plane for the k-grid, e.g., "xy", "yz", "zx".
            k_range_x: Tuple (min_kx, max_kx) for the first dimension of the plane.
            k_range_y: Tuple (min_ky, max_ky) for the second dimension of the plane.
            n_kx: Number of k-points along the first dimension.
            n_ky: Number of k-points along the second dimension.
            k_fixed_val: Value of the k-component perpendicular to the plane (e.g., kz for "xy" plane).

        Returns:
            Tuple of (k_magnitudes, k_vectors_3d, k_grid_shape).
            k_magnitudes: An empty 1D numpy array (as magnitudes are not well-defined for a grid point by point).
            k_vectors_3d: 2D array of 3D k-vectors (2π/Å), shape (n_kx * n_ky, 3).
            k_grid_shape: Tuple (n_kx, n_ky) indicating the grid dimensions.
        """
        if n_kx <= 0 or n_ky <= 0:
            raise ValueError("Number of k-points (n_kx, n_ky) must be positive.")

        kx_vals = np.linspace(k_range_x[0], k_range_x[1], n_kx, dtype=np.float32)
        ky_vals = np.linspace(k_range_y[0], k_range_y[1], n_ky, dtype=np.float32)

        k_vectors_list = []
        if plane.lower() == "xy":
            for kx in kx_vals:
                for ky in ky_vals:
                    k_vectors_list.append([kx, ky, k_fixed_val])
        elif plane.lower() == "yz":
            for ky in kx_vals: # Note: kx_vals corresponds to the first range, which is y for yz plane
                for kz in ky_vals: # Note: ky_vals corresponds to the second range, which is z for yz plane
                    k_vectors_list.append([k_fixed_val, ky, kz])
        elif plane.lower() == "zx":
            for kz in kx_vals: # Note: kx_vals corresponds to the first range, which is z for zx plane
                for kx in ky_vals: # Note: ky_vals corresponds to the second range, which is x for zx plane
                    k_vectors_list.append([kx, k_fixed_val, kz])
        else:
            raise ValueError(f"Invalid plane specified: {plane}. Must be 'xy', 'yz', or 'zx'.")

        k_vectors_3d = np.array(k_vectors_list, dtype=np.float32)
        k_grid_shape = (n_kx, n_ky)
        
        # For k_points (1D representation), return an empty array as it's not directly applicable here.
        # The main k-point information is in k_vectors_3d and k_grid_shape.
        k_mags_for_grid = np.array([], dtype=np.float32) 
        return k_mags_for_grid, k_vectors_3d, k_grid_shape

    def calculate(self, k_points_mags: np.ndarray, k_vectors_3d: np.ndarray,
                  basis_atom_indices: Optional[Union[List[int], List[List[int]], np.ndarray]] = None,
                  basis_atom_types: Optional[Union[List[int], List[List[int]]]] = None,
                  summation_mode: str = 'coherent',
                  k_grid_shape: Optional[Tuple[int, int]] = None,
                  k_chunk_size: int = 500
                  ) -> SED:
        
        if summation_mode not in ['coherent', 'incoherent']:
            raise ValueError(f"summation_mode must be 'coherent' or 'incoherent', got {summation_mode}")

        n_t, n_atoms_tot = self.traj.n_frames, self.traj.n_atoms
        if n_t == 0 or n_atoms_tot == 0:
            logger.warning("Cannot calculate SED: 0 frames or 0 atoms.")
            # Return an empty SED object
            return SED(np.array([],dtype=np.complex64).reshape(0,0,3), 
                       np.array([],dtype=np.float32), 
                       k_points_mags, 
                       k_vectors_3d,
                       k_grid_shape=k_grid_shape,
                       is_complex=True, 
                       phase=None)

        mean_pos_all = np.mean(self.traj.positions, axis=0, dtype=np.float32)
        freqs = np.fft.fftfreq(n_t, d=self.dt_ps) if n_t > 0 else np.array([],dtype=np.float32)

        # Determine atom groups for SED calculation
        atom_groups: List[np.ndarray] = []

        if basis_atom_types is not None:
            if basis_atom_indices is not None:
                logger.warning("Both basis_atom_types and basis_atom_indices provided. Using basis_atom_types.")
            
            processed_basis_atom_types: List[List[int]] = []
            if isinstance(basis_atom_types, list) and len(basis_atom_types) > 0:
                if all(isinstance(item, list) for item in basis_atom_types):
                    processed_basis_atom_types = basis_atom_types
                elif all(isinstance(item, int) for item in basis_atom_types):
                    if summation_mode == 'incoherent': 
                        processed_basis_atom_types = [[t] for t in basis_atom_types]
                    else: 
                        processed_basis_atom_types = [list(basis_atom_types)]
                else:
                    raise ValueError("basis_atom_types must be a list of ints or a list of lists of ints.")
            elif isinstance(basis_atom_types, int):
                 processed_basis_atom_types = [[basis_atom_types]]

            for type_group in processed_basis_atom_types:
                indices = np.where(np.isin(self.traj.types, type_group))[0]
                if indices.size > 0:
                    atom_groups.append(indices)
                else:
                    logger.warning(f"No atoms found for type group {type_group}. Skipping.")
        
        elif basis_atom_indices is not None:
            processed_basis_atom_indices: List[np.ndarray] = []
            if isinstance(basis_atom_indices, list):
                if len(basis_atom_indices) == 0:
                    pass 
                elif all(isinstance(item, list) for item in basis_atom_indices): 
                    for sublist in basis_atom_indices:
                        arr = np.asarray(sublist, dtype=int)
                        if arr.size > 0 : processed_basis_atom_indices.append(arr)
                elif all(isinstance(item, int) for item in basis_atom_indices): 
                    arr = np.asarray(basis_atom_indices, dtype=int)
                    if arr.size > 0: processed_basis_atom_indices.append(arr)
                else:
                    raise ValueError("basis_atom_indices must be a list of ints or a list of lists of ints.")
            elif isinstance(basis_atom_indices, np.ndarray):
                if basis_atom_indices.ndim == 1 and basis_atom_indices.size > 0:
                     processed_basis_atom_indices.append(basis_atom_indices.astype(int))
                else:
                    logger.warning("Unsupported np.ndarray format for basis_atom_indices. Using all atoms if no other basis defined.")
            
            for grp_idx in processed_basis_atom_indices:
                if np.any(grp_idx >= n_atoms_tot) or np.any(grp_idx < 0):
                    raise ValueError("Atom indices in basis out of bounds.")
                if grp_idx.size > 0:
                    atom_groups.append(grp_idx)

        if not atom_groups:
            logger.debug(f"No specific basis provided or basis resulted in empty groups. Using all {n_atoms_tot} atoms as a single group.")
            atom_groups.append(np.arange(n_atoms_tot))
            if summation_mode == 'incoherent' and n_atoms_tot > 0:
                logger.info("Using all atoms. Incoherent sum will effectively be a coherent sum of all atoms.")
        
        # --- K-vector chunking logic ---
        num_k_vectors = len(k_vectors_3d)
        # Ensure k_chunk_size is at least 1 and not larger than total k-vectors
        actual_k_chunk_size = min(max(1, k_chunk_size), num_k_vectors) if num_k_vectors > 0 else 1
        num_chunks = (num_k_vectors + actual_k_chunk_size - 1) // actual_k_chunk_size if num_k_vectors > 0 else 0

        # Initialize the full sed_data array based on summation_mode
        is_complex_output: bool
        if summation_mode == 'coherent' or len(atom_groups) <= 1:
            full_sed_data = np.zeros((len(freqs), num_k_vectors, 3), dtype=np.complex64)
            is_complex_output = True
        else: # incoherent
            full_sed_data = np.zeros((len(freqs), num_k_vectors), dtype=np.float32) # Store sum of intensities
            is_complex_output = False

        if num_k_vectors == 0: # Handle empty k_vectors_3d
            logger.warning("k_vectors_3d is empty. Returning SED object with empty SED data.")
            # Fall through to SED object creation with empty/zero data

        for i_chunk in range(num_chunks):
            start_idx = i_chunk * actual_k_chunk_size
            end_idx = min((i_chunk + 1) * actual_k_chunk_size, num_k_vectors)
            current_k_vectors_chunk = k_vectors_3d[start_idx:end_idx]
            
            if current_k_vectors_chunk.shape[0] == 0: continue # Skip if chunk is empty

            logger.debug(f"Processing k-chunk {i_chunk+1}/{num_chunks} (indices {start_idx}-{end_idx-1})")

            if is_complex_output: # Coherent summation
                if len(atom_groups) > 1:
                    final_group_indices = np.unique(np.concatenate(atom_groups)).astype(int)
                elif len(atom_groups) == 1:
                    final_group_indices = atom_groups[0]
                else: # Should not happen due to fallback
                    final_group_indices = np.array([], dtype=int)

                if final_group_indices.size == 0:
                    logger.warning(f"Final atom group for SED k-chunk {i_chunk+1} is empty. SED chunk will be zero.")
                    # full_sed_data is already zeros
                    continue
                
                logger.debug(f"  Calculating SED coherently for {len(final_group_indices)} atoms for k-chunk {i_chunk+1}.")
                sed_chunk_data = self._calculate_sed_for_group(current_k_vectors_chunk, final_group_indices, mean_pos_all)
                full_sed_data[:, start_idx:end_idx, :] = sed_chunk_data
            
            else: # Incoherent summation
                logger.debug(f"  Calculating SED incoherently for {len(atom_groups)} groups for k-chunk {i_chunk+1}.")
                # Accumulator for summed intensities for the current chunk (real, positive)
                summed_intensity_chunk = np.zeros((len(freqs), current_k_vectors_chunk.shape[0]), dtype=np.float32)
                
                for i_grp, grp_indices in enumerate(atom_groups):
                    if grp_indices.size == 0:
                        logger.debug(f"    Skipping empty atom group {i_grp+1} in k-chunk {i_chunk+1}.")
                        continue
                    logger.debug(f"    Calculating for group {i_grp+1}/{len(atom_groups)} with {len(grp_indices)} atoms in k-chunk {i_chunk+1}.")
                    sed_group_complex_chunk = self._calculate_sed_for_group(current_k_vectors_chunk, grp_indices, mean_pos_all)
                    # Sum of magnitudes squared for incoherent sum, sum over polarizations
                    summed_intensity_chunk += np.sum(np.abs(sed_group_complex_chunk)**2, axis=-1) 
                
                full_sed_data[:, start_idx:end_idx] = summed_intensity_chunk
        
        # Construct and return SED object
        return SED(full_sed_data, 
                   freqs, 
                   k_points_mags, 
                   k_vectors_3d, # Store the original full k_vectors
                   k_grid_shape=k_grid_shape,
                   is_complex=is_complex_output, 
                   phase=None) # Phase is not calculated in this method

    def calculate_chiral_phase(self, Z1: np.ndarray, Z2: np.ndarray, angle_range_opt: str = "C") -> np.ndarray:
        if Z1.shape != Z2.shape: 
            raise ValueError("Z1 and Z2 shapes must match for chiral phase.")
        if Z1.size == 0: 
            return np.array([], dtype=np.float32).reshape(Z1.shape)

        if angle_range_opt == "C": 
            p1, p2 = np.angle(Z1), np.angle(Z2)
            delta_p = p1 - p2
            delta_p = (delta_p + np.pi) % (2*np.pi) - np.pi # Wrap to [-pi, pi]
            delta_p[delta_p > (np.pi/2)] = np.pi - delta_p[delta_p > (np.pi/2)]   # Fold Q2
            delta_p[delta_p < (-np.pi/2)] = -np.pi - delta_p[delta_p < (-np.pi/2)] # Fold Q3
            return delta_p.astype(np.float32)
        else: 
            nw, nk = Z1.shape
            out_phase = np.zeros((nw,nk), dtype=np.float32)
            for i in range(nw):
                for j in range(nk):
                    v1r,v1i = Z1[i,j].real, Z1[i,j].imag
                    v2r,v2i = Z2[i,j].real, Z2[i,j].imag
                    m1sq,m2sq = v1r**2+v1i**2, v2r**2+v2i**2
                    if m1sq<1e-18 or m2sq<1e-18: 
                        angle=0.0
                    else:
                        m1,m2 = np.sqrt(m1sq), np.sqrt(m2sq)
                        if angle_range_opt == "A": 
                            angle = np.arccos(np.clip((v1r*v2r+v1i*v2i)/(m1*m2), -1.0, 1.0))
                        elif angle_range_opt == "B": 
                            angle = np.arcsin(np.clip((v1r*v2i-v1i*v2r)/(m1*m2), -1.0, 1.0))
                        else: 
                            logger.warning(f"Unknown angle_range_opt '{angle_range_opt}'. Angle=0.")
                            angle=0.0
                    out_phase[i,j] = angle
            return out_phase

    def ised(self, k_dir_spec: Union[str, int, float, List[float], np.ndarray, Dict[str,float]],
             k_target: float, w_target: float, char_len_k_path: float,
             nk_on_path: int = 100, bz_cov_ised: float = 1.0,
             basis_atom_idx_ised: Optional[List[int]] = None, 
             basis_atom_types_ised: Optional[List[int]] = None,
             rescale_factor: Union[str, float] = 1.0, n_recon_frames: int = 100,
             dump_filepath: str = "iSED_reconstruction.dump",
             plot_dir_ised: Optional[Path] = None, plot_max_freq: Optional[float] = None,
             plot_theme: str = 'light'
             ) -> None:
        logger.info("Starting iSED reconstruction.")
        avg_pos = np.mean(self.traj.positions, axis=0, dtype=np.float32)
        sys_atom_types = self.traj.types.astype(int)
        n_atoms_total = self.traj.n_atoms
        k_dir_unit = parse_direction(k_dir_spec)

        recon_atom_groups: List[np.ndarray] = []
        if basis_atom_idx_ised and len(basis_atom_idx_ised) > 0:
            if isinstance(basis_atom_idx_ised[0], list): 
                logger.info(f"iSED using specified atom index groups: {len(basis_atom_idx_ised)} groups.")
                for grp_idx in basis_atom_idx_ised:
                    grp_arr = np.asarray(grp_idx, dtype=int)
                    if np.any(grp_arr >= n_atoms_total) or np.any(grp_arr < 0): 
                        raise ValueError(f"Atom indices in group {grp_idx} out of bounds.")
                    if grp_arr.size > 0: 
                        recon_atom_groups.append(grp_arr)
            else: 
                logger.info(f"iSED using single atom index group ({len(basis_atom_idx_ised)} atoms).")
                grp_arr = np.asarray(basis_atom_idx_ised, dtype=int)
                if np.any(grp_arr >= n_atoms_total) or np.any(grp_arr < 0): 
                    raise ValueError("Atom indices out of bounds.")
                if grp_arr.size > 0: 
                    recon_atom_groups.append(grp_arr)
            if basis_atom_types_ised and len(basis_atom_types_ised) > 0:
                logger.warning("iSED: atom_indices and atom_types provided. Using atom_indices.")
        elif basis_atom_types_ised and len(basis_atom_types_ised) > 0:
            if isinstance(basis_atom_types_ised[0], list): 
                logger.info(f"iSED using specified atom type groups: {len(basis_atom_types_ised)} groups.")
                for type_grp in basis_atom_types_ised:
                    mask = np.isin(sys_atom_types, type_grp)
                    grp_idx = np.where(mask)[0]
                    if grp_idx.size > 0: 
                        recon_atom_groups.append(grp_idx)
                    else: 
                        logger.warning(f"No atoms for type group {type_grp} in iSED.")
            else: 
                logger.info(f"iSED using each atom type as a group for types: {basis_atom_types_ised}.")
                for atom_type_val in basis_atom_types_ised:
                    mask = np.isin(sys_atom_types, [atom_type_val])
                    grp_idx = np.where(mask)[0]
                    if grp_idx.size > 0: 
                        recon_atom_groups.append(grp_idx)
                    else: 
                        logger.warning(f"No atoms for type {atom_type_val} in iSED.")
        else:
            logger.info("iSED using all atoms as a single group.")
            recon_atom_groups.append(np.arange(n_atoms_total))

        if not recon_atom_groups: 
            logger.error("iSED: No atom groups for reconstruction. Aborting.")
            return

        logger.debug(f"iSED k-path: dir={k_dir_spec}, L_char={char_len_k_path}, nk={nk_on_path}, bz_cov={bz_cov_ised}")
        k_mags_ised, k_vecs_ised = self.get_k_path(direction_spec=k_dir_unit, bz_coverage=bz_cov_ised,
                                                 n_k=nk_on_path, lat_param=char_len_k_path)
        
        wiggles = np.zeros((n_recon_frames, n_atoms_total, 4), dtype=np.float32) # x,y,z,type
        time_p = np.linspace(0, 2*np.pi, n_recon_frames, endpoint=False)
        pos_proj_k_dir = np.dot(avg_pos, k_dir_unit)

        k_match_idx = np.argmin(np.abs(k_mags_ised - k_target))
        k_actual = k_mags_ised[k_match_idx]
        logger.info(f"iSED: Target k={k_target:.4f} -> Matched k={k_actual:.4f} (2π/Å, idx {k_match_idx})")

        recon_done, max_wiggle_amp_all = False, 0.0
        std_dev_sum, n_atoms_recon_sum = 0.0, 0
        ised_input_intensity_plot, ised_input_freqs_plot = None, None

        for i_grp, grp_atom_idx in enumerate(recon_atom_groups):
            if grp_atom_idx.size == 0: 
                logger.debug(f"Skipping empty iSED group {i_grp+1}.")
                continue
            
            grp_types_str = np.unique(sys_atom_types[grp_atom_idx])
            logger.info(f"iSED Group {i_grp+1}/{len(recon_atom_groups)}: {len(grp_atom_idx)} atoms (types: {grp_types_str}).")
            logger.debug(f"  iSED Group {i_grp+1}: Calculating SED for {len(grp_atom_idx)} atoms.")
            sed_object_group = self.calculate(k_points_mags=k_mags_ised, 
                                            k_vectors_3d=k_vecs_ised, 
                                            basis_atom_indices=grp_atom_idx,
                                            k_grid_shape=None, # k-path, so no grid shape
                                            summation_mode='coherent') # iSED typically processes groups coherently first
            
            sed_group_data = sed_object_group.sed
            freqs_group = sed_object_group.freqs
            # is_complex = sed_object_group.is_complex # Not strictly needed for current logic
            
            if ised_input_freqs_plot is None: 
                ised_input_freqs_plot = freqs_group
            elif not np.array_equal(ised_input_freqs_plot, freqs_group): 
                logger.warning("iSED group freq arrays differ. Plotting may be inconsistent.")

            grp_intensity = np.sum(np.abs(sed_group_data)**2, axis=-1)
            if ised_input_intensity_plot is None: 
                ised_input_intensity_plot = grp_intensity.copy()
            else:
                if ised_input_intensity_plot.shape == grp_intensity.shape: 
                    ised_input_intensity_plot += grp_intensity
                else: 
                    logger.warning(f"iSED group intensity shape mismatch (group {i_grp+1}). Skipping accumulation.")

            w_match_idx = np.argmin(np.abs(freqs_group - w_target))
            w_actual = freqs_group[w_match_idx]
            logger.info(f"  iSED Group {i_grp+1}: Target ω={w_target:.3f} -> Matched ω={w_actual:.3f} (THz, idx {w_match_idx})")

            if grp_atom_idx.size > 0:
                logger.debug(f"  DEBUG Group {i_grp+1}: Spatial phase for k={k_actual:.4f} (rad/Å):")
                for atom_sys_idx in grp_atom_idx[:min(5, len(grp_atom_idx))]: 
                    r_proj = pos_proj_k_dir[atom_sys_idx]
                    spatial_p = k_actual * r_proj
                    logger.debug(f"    Atom SysIdx={atom_sys_idx}, Type={sys_atom_types[atom_sys_idx]}, r_proj={r_proj:.4f}Å, k*r_proj={spatial_p:.4f}rad")

            for pol_ax in range(3):
                sed_pol_data = sed_group_data[:, :, pol_ax]
                complex_amp_grp = sed_pol_data[w_match_idx, k_match_idx]
                proj_pos_grp = pos_proj_k_dir[grp_atom_idx]
                recon_motion_comp = np.real(complex_amp_grp * np.exp(1j * time_p[:,None] - 1j * k_actual * proj_pos_grp[None,:]))
                wiggles[:, grp_atom_idx, pol_ax] += recon_motion_comp
            
            recon_done = True
            if isinstance(rescale_factor, str) and rescale_factor.lower() == "auto":
                max_amp_grp = np.amax(np.abs(wiggles[:, grp_atom_idx, :3])) if grp_atom_idx.size > 0 else 0.0
                max_wiggle_amp_all = max(max_wiggle_amp_all, max_amp_grp)
                if grp_atom_idx.size > 0:
                    orig_disp_grp = self.traj.positions[:, grp_atom_idx,:] - avg_pos[None, grp_atom_idx,:]
                    std_dev_sum += np.std(orig_disp_grp) * len(grp_atom_idx)
                    n_atoms_recon_sum += len(grp_atom_idx)

        if not recon_done: 
            logger.error("iSED: No reconstruction performed (empty atom groups?).")
            return

        wiggles[0,:,3] = sys_atom_types # Store types in 4th component of 1st frame
        all_recon_idx = np.unique(np.concatenate(recon_atom_groups)) if recon_atom_groups and any(g.size > 0 for g in recon_atom_groups) else np.array([])
        
        if all_recon_idx.size > 0:
            if isinstance(rescale_factor, str) and rescale_factor.lower() == "auto":
                if max_wiggle_amp_all > 1e-9:
                    wiggles[:,all_recon_idx,:3] /= max_wiggle_amp_all 
                    avg_std_dev_disp = std_dev_sum / n_atoms_recon_sum if n_atoms_recon_sum > 0 else 0.0
                    if avg_std_dev_disp > 1e-9: 
                        wiggles[:,all_recon_idx,:3] *= avg_std_dev_disp
                    logger.info(f"iSED: Auto-rescaled. Max amp: {max_wiggle_amp_all:.3e}, Avg StdDev scale: {avg_std_dev_disp:.3e}")
                else: 
                    logger.warning("iSED: Max wiggle amp near zero. Auto-rescaling ineffective.")
            elif isinstance(rescale_factor, (int, float)):
                wiggles[:,all_recon_idx,:3] *= rescale_factor
                logger.info(f"iSED: Rescaled wiggles by factor {rescale_factor}.")
        else: 
            logger.warning("iSED: No atoms reconstructed, skipping rescaling.")

        final_pos_dump = avg_pos[None,:,:] + wiggles[:,:,:3]
        atom_types_dump = wiggles[0,:,3].astype(int) # Ensure types are integer for dump
        
        # Pass the full box_matrix for correct triclinic box representation
        out_to_qdump(dump_filepath, final_pos_dump, atom_types_dump, self.traj.box_matrix)
        logger.info(f"iSED reconstruction saved: {dump_filepath}")

        if plot_dir_ised and ised_input_intensity_plot is not None and ised_input_freqs_plot is not None:
            logger.info("Plotting iSED input spectrum (incoherently summed groups).")
            ised_mock_sed_plot = np.zeros((*ised_input_intensity_plot.shape, 3), dtype=np.complex64)
            ised_mock_sed_plot[:,:,0] = np.sqrt(ised_input_intensity_plot + 1e-20) # Store in first pol for SED object structure
            ised_plot_obj = SED(sed=ised_mock_sed_plot, freqs=ised_input_freqs_plot,
                                k_points=k_mags_ised, k_vectors=k_vecs_ised,
                                is_complex=True) # Mock SED is technically complex here, intensity handled by plotter
            
            # --- Filename Generation --- 
            k_dir_str = ""
            if isinstance(k_dir_spec, str):
                k_dir_str = k_dir_spec.replace(" ","_").replace("/", "-")
            elif isinstance(k_dir_spec, (list, tuple, np.ndarray)):
                arr = np.asarray(k_dir_spec)
                k_dir_str = f"({','.join([f'{x:.2f}' for x in arr])})"
            elif isinstance(k_dir_spec, dict):
                k_dir_str = f"(h{k_dir_spec.get('h',0)}_k{k_dir_spec.get('k',0)}_l{k_dir_spec.get('l',0)})"
            else:
                k_dir_str = str(k_dir_spec)
            k_dir_str = k_dir_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "") # Clean common brackets

            k_target_str = f"{k_target:.2f}".replace('.','p')
            w_target_str = f"{w_target:.2f}".replace('.','p')
            
            # New filename pattern:
            ised_plot_fname = plot_dir_ised / f"iSED_{k_dir_str}_{k_target_str}_{w_target_str}.png"
            # Old filename pattern for reference (was more complex):
            # ised_plot_fname = plot_dir_ised / f"ised_input_sed_{ised_k_dir_label}_k{k_str}_w{w_str}_inc_sum.png"
            
            w_match_plot_idx = np.argmin(np.abs(ised_input_freqs_plot - w_target))
            w_actual_plot = ised_input_freqs_plot[w_match_plot_idx]
            hl_info = {'k_point_target':k_actual, 'freq_point_target':w_actual_plot}
            
            max_freq_ised_plot = plot_max_freq
            if max_freq_ised_plot is None and ised_input_freqs_plot.size > 0:
                 max_freq_ised_plot = np.max(ised_input_freqs_plot)
            
            plot_args_ised = {
                'title': f"Summed iSED Input Spectrum (k≈{k_actual:.3f}, ω≈{w_actual_plot:.3f})",
                'direction_label': k_dir_str, # Use the formatted k_dir_str for label consistency
                'highlight_region': hl_info,
                'max_freq': max_freq_ised_plot,
                'intensity_scale': 'sqrt',
                'theme': plot_theme  # Pass theme to SEDPlotter
            }
            SEDPlotter(ised_plot_obj, '2d_intensity', str(ised_plot_fname), **plot_args_ised).generate_plot()
            logger.info(f"iSED input spectrum plot saved: {ised_plot_fname.name}")
        elif plot_dir_ised:
            logger.warning("iSED plot requested, but no combined SED data available.")



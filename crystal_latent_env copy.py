# crystal_latent_env.py
import os
import pickle
from typing import Optional, Callable, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

from pymatgen.core import Lattice, Structure

from models import VAE, VAE_Lattice
from dataset import MyDataset
from data_utils import (
    minmax_,
    inv_minmax,
    data_query,
    FTCP_represent,
    pad,
    convert_cif,
)
import joblib
import xgboost as xgb

## import to run csllm with another environment
import subprocess
import sys


## import CV modules
from working_VAE_CV.data_prep_cv_vae_pred_head import generate_and_save
from working_VAE_CV.vae_infer_batch import infer_VAE


class CrystalLatentEnv(gym.Env):
    """
    Gym-style environment for RL on the Crystal-LSBO latent space.

    - Internal state: a latent vector z in the Combined-VAE space.
    - Observation: a "measurement plot" derived from the decoded structure.
      Here we use the FTCP representation slice (0:185, 0:4) flattened.
    - Actions: discrete +/- step in each dimension of z.
      action i:
         dim   = i // 2
         sign  = +1 if i % 2 == 0 else -1
      so action_space.n = 2 * combined_z_size
    - Reward: scalar per state, default = - formation_energy predicted by
      the XGBoost black-box model (higher reward = more stable crystal).

    You can optionally override the measurement/reward with a callback:
        measurement_fn(ftcp: np.ndarray, info: dict) -> (obs, reward)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        combined_z_size: int,
        ckpt_name: str,
        exp_name: str = "crystal_lsbo_repo",
        mp_api_key: str = "",
        step_size: float = 0.1,
        bound: float = 3.0,
        max_steps: int = 10,
        measurement_fn: Optional[
            Callable[[np.ndarray, Dict], Tuple[np.ndarray, float]]
        ] = None,
        device: Optional[torch.device] = None,
        use_black_box_reward: bool = True,
        save_cif: bool = False,
    ):
        """
        Args
        ----
        combined_z_size : int
            Dimension of the Combined-VAE latent space.
        ckpt_name : str
            Checkpoint filename for the Combined-VAE (e.g. 'combined_vae.pt').
        exp_name : str
            Folder containing pre-trained models and artifacts (default: 'crystal_lsbo_repo').
        mp_api_key : str
            Materials Project API key (only needed if data.csv / FTCP_representation.pkl
            are not already present).
        step_size : float
            Magnitude of latent step per action.
        bound : float
            Latent coordinates are clipped to [-bound, bound].
        max_steps : int
            Episode length before 'done'.
        measurement_fn : callable or None
            Optional callback to override observation & reward given FTCP and info.
        device : torch.device or None
            Torch device. If None, tries CUDA then CPU.
        use_black_box_reward : bool
            If True, use XGBoost FE predictor for reward. If False, reward = 0 unless
            measurement_fn returns something else.
        save_cif : bool
            If True, convert each decoded structure to CIF (expensive I/O).
        """
        super().__init__()

        self._episode_id=0  ##added
        self.combined_z_size = combined_z_size
        self.ckpt_name = ckpt_name
        self.exp_name = exp_name
        self.exp_name2 = "crystal_rl_env"
        self.mp_api_key = mp_api_key
        self.step_size = step_size
        self.bound = bound
        self.max_steps = max_steps
        self.measurement_fn = measurement_fn
        self.use_black_box_reward = use_black_box_reward
        self.save_cif = save_cif

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        # Crystal-LSBO constants (mirror repo defaults)
        self.element_z_size = 16
        self.coord_z_size = 16
        self.lattice_z_size = 3
        self.max_elms = 3
        self.min_elms = 1
        self.max_sites = 10
        self.property_for_predict = ["formation_energy_per_atom", "band_gap"]

        # --------------------------
        # Load / prepare data & models
        # --------------------------
        self._init_data_and_models()

        # --------------------------
        # RL spaces
        # --------------------------
        # Two actions per dimension: +step or -step
        self.action_space = spaces.Discrete(2 * self.combined_z_size)

        # Observation is FTCP slice (185, 4) flattened → 740-d vector.
        # Values can be quite broad, so keep bounds wide.
        self.obs_dim = 185 * 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Internal state
        self._z = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Data & model initialisation (mostly adapted from random_experiment.py)
    # ------------------------------------------------------------------
    def _init_data_and_models(self):
        # ---- Load or query data.csv ----
        data_csv_path = os.path.join(self.exp_name, "data.csv")
        if os.path.exists(data_csv_path):
            dataframe = pd.read_csv(data_csv_path)
        else:
            # This requires a valid Materials Project API key.
            dataframe = data_query(
                self.mp_api_key,
                max_elms=self.max_elms,
                min_elms=self.min_elms,
                max_sites=self.max_sites,
            )
            os.makedirs(self.exp_name, exist_ok=True)
            dataframe.to_csv(data_csv_path, index=False)

        # ---- FTCP representation ----
        ftcp_path = os.path.join(self.exp_name, "FTCP_representation.pkl")
        nsites_path = os.path.join(self.exp_name, "Nsites.pkl")
        if os.path.exists(ftcp_path) and os.path.exists(nsites_path):
            with open(ftcp_path, "rb") as f:
                FTCP_representation = pickle.load(f)
            with open(nsites_path, "rb") as f:
                Nsites = pickle.load(f)
        else:
            FTCP_representation, Nsites = FTCP_represent(
                dataframe,
                max_elms=self.max_elms,
                max_sites=self.max_sites,
                return_Nsites=True,
            )
            FTCP_representation = pad(FTCP_representation, 2)
            os.makedirs(self.exp_name, exist_ok=True)
            with open(ftcp_path, "wb") as f:
                pickle.dump(FTCP_representation, f)
            with open(nsites_path, "wb") as f:
                pickle.dump(Nsites, f)

        # Pad again like in original code
        FTCP_representation = pad(FTCP_representation, 2)
        self.X_array_origin = FTCP_representation
        self.Y_array = dataframe[self.property_for_predict].values
        del dataframe

        # torch.manual_seed(42)
        # np.random.seed(42)

        # Subsets (same slicing as repo)
        subset_1 = self.X_array_origin[:, 0:103, :4]  # element
        subset_2 = self.X_array_origin[:, 103:104, :3]  # angle
        subset_3 = self.X_array_origin[:, 104:105, :3]  # abc
        subset_4 = self.X_array_origin[:, 105:145, :4]  # coords
        subset_5 = self.X_array_origin[:, 145:185, :4]  # occupancy

        # ---------------- Element model ----------------
        X_array_element = np.concatenate((subset_1, subset_5), axis=1)
        (
            X_normed_element,
            Y_normed_element,
            self.scaler_x_element,
            self.scaler_y_element,
        ) = minmax_(X_array_element, self.Y_array)
        input_X_full_element = torch.tensor(
            X_normed_element.transpose(0, 2, 1)
        ).double()
        input_Y_element = torch.tensor(Y_normed_element).double()
        # Element VAE input (as in repo: pad to feature 144)
        self.input_X_element = F.pad(
            input_X_full_element[:, :4, :143], (0, 1), "constant", 0
        )
        self.model_element = VAE(
            self.element_z_size, self.input_X_element, input_Y_element
        ).to(self.device).to(torch.float64)
        self.model_element.load_state_dict(
            torch.load(
                os.path.join(self.exp_name, "element_vae.pt"),
                map_location=self.device,
            )
        )

        # Additional scaler for element+occ reconstruction to original scale
        X_elem_occ = np.concatenate(
            (self.X_array_origin[:, 0:103, :4], self.X_array_origin[:, 145:185, :4]),
            axis=1,
        )
        (
            _,
            self.Y_normed_elemocc,
            self.scaler_x_elemocc,
            self.scaler_y_elemocc,
        ) = minmax_(X_elem_occ, self.Y_array)

        # ---------------- Lattice model ----------------
        X_array_lattice = np.concatenate((subset_2, subset_3), axis=1)
        X_array_lattice = X_array_lattice[:, :, :4]
        (
            X_normed_lattice,
            Y_normed_lattice,
            self.scaler_x_lattice,
            self.scaler_y_lattice,
        ) = minmax_(X_array_lattice, self.Y_array)
        input_X_full_lattice = torch.tensor(
            X_normed_lattice.transpose(0, 2, 1)
        ).double()
        input_Y_lattice = torch.tensor(Y_normed_lattice).double()
        self.input_X_lattice = input_X_full_lattice[:, :4, :]
        self.Y_normed_lattice = Y_normed_lattice
        self.model_lattice = VAE_Lattice(
            self.lattice_z_size, self.input_X_lattice, input_Y_lattice
        ).to(self.device).to(torch.float64)
        self.model_lattice.load_state_dict(
            torch.load(
                os.path.join(self.exp_name, "lattice_vae.pt"),
                map_location=self.device,
            )
        )

        # ---------------- Coordinate model ----------------
        X_array_coord = subset_4
        (
            X_normed_coord,
            Y_normed_coord,
            self.scaler_x_coord,
            self.scaler_y_coord,
        ) = minmax_(X_array_coord, self.Y_array)
        input_X_full_coord = torch.tensor(
            X_normed_coord.transpose(0, 2, 1)
        ).double()
        input_Y_coord = torch.tensor(Y_normed_coord).double()
        self.input_X_coord = input_X_full_coord[:, :4, :]
        self.Y_normed_coord = Y_normed_coord
        self.model_coord = VAE(
            self.coord_z_size, self.input_X_coord, input_Y_coord
        ).to(self.device).to(torch.float64)
        self.model_coord.load_state_dict(
            torch.load(
                os.path.join(self.exp_name, "coordinate_vae.pt"),
                map_location=self.device,
            )
        )

        # ---------------- Combined model ----------------
        input_X_comb = torch.load(
            os.path.join(self.exp_name, "combined_vae_input_latent_data.pt")
        )
        self.data_min = input_X_comb.min()
        self.data_max = input_X_comb.max()
        input_X_comb = F.pad(input_X_comb, (0, 1), "constant", 0)
        input_X_comb = input_X_comb.view(input_X_comb.shape[0], 3, 12)

        # Just need shapes & Y tensor for model construction
        input_Y_comb = input_Y_element  # any Y tensor of correct shape works
        self.model_combined = VAE(
            self.combined_z_size, input_X_comb, input_Y_comb
        ).to(self.device).to(torch.float64)
        self.model_combined.load_state_dict(
            torch.load(
                os.path.join(self.exp_name, self.ckpt_name),
                map_location=self.device,
            )
        )

        # ---------------- Black-box property model ----------------
        self.xgb_model = None
        if self.use_black_box_reward:
            xgb_path = os.path.join(self.exp_name, "xgb_black_box.pkl")
            if os.path.exists(xgb_path):
                with open(xgb_path, "rb") as f:
                    self.xgb_model = pickle.load(f)
            else:
                print(
                    f"[CrystalLatentEnv] Warning: {xgb_path} not found. "
                    "Reward will be 0 unless measurement_fn is provided."
                )

        # Pre-load element list if we want CIFs
        self.elm_str = None
        if self.save_cif:
            self.elm_str = joblib.load("data/element.pkl")

    # ------------------------------------------------------------------
    # Decoding helpers (adapted from target_property_wrapper in repo)
    # ------------------------------------------------------------------
    def _rescale_and_divide(self, x_recon: torch.Tensor):
        """
        Undo Combined-VAE scaling and split into Element / Lattice / Coord latents.
        """
        x_recon = x_recon.view(-1, 36)
        x_recon = x_recon * (self.data_max - self.data_min) + self.data_min
        element_z = x_recon[:, : self.element_z_size]
        lattice_z = x_recon[:, self.element_z_size : (self.element_z_size + 3)]
        coord_z = x_recon[
            :,
            (self.element_z_size + 3) : (self.element_z_size + 3 + self.coord_z_size),
        ]
        return element_z, lattice_z, coord_z

    # def _get_lattice_generation(self, lattice_z_sample_first: torch.Tensor):
    #     """
    #     Decode lattice latent → lattice features in original scale.
    #     """
    #     lattice_z_sample = self.model_lattice.decoder.forward(
    #         lattice_z_sample_first.to(self.device).to(torch.float64)
    #     )
    #     lattice_z_sample = torch.tensor(lattice_z_sample).to(self.device)
    #     lattice_z_sample = lattice_z_sample.detach().cpu().numpy()

    #     # Use a dummy slice of normalized Y for inverse scaling
    #     dummy_Y = self.Y_normed_lattice[: lattice_z_sample.shape[0]]
    #     lattice_z_sample, _ = inv_minmax(
    #         lattice_z_sample,
    #         dummy_Y,
    #         self.scaler_x_lattice,
    #         self.scaler_y_lattice,
    #     )
    #     return lattice_z_sample

    # def _get_coord_generation(self, coord_z_sample_first: torch.Tensor):
    #     """
    #     Decode coordinate latent → coordinate features in original scale.
    #     """
    #     coord_z_sample = self.model_coord.decoder.forward(
    #         coord_z_sample_first.to(self.device).to(torch.float64)
    #     )
    #     coord_z_sample = torch.tensor(coord_z_sample).to(self.device)
    #     coord_z_sample = coord_z_sample.detach().cpu().numpy()

    #     dummy_Y = self.Y_normed_coord[: coord_z_sample.shape[0]]
    #     coord_z_sample, _ = inv_minmax(
    #         coord_z_sample, dummy_Y, self.scaler_x_coord, self.scaler_y_coord
    #     )
    #     return coord_z_sample

    def _get_lattice_generation(self, lattice_z_sample_first: torch.Tensor):
        """
        Decode lattice latent → lattice features in original scale.
        """
        lattice_z_sample = self.model_lattice.decoder.forward(
            lattice_z_sample_first.to(self.device).to(torch.float64)
        )
        # lattice_z_sample is already a tensor on the right device
        lattice_z_sample = lattice_z_sample.detach().cpu().numpy()

        dummy_Y = self.Y_normed_lattice[: lattice_z_sample.shape[0]]
        lattice_z_sample, _ = inv_minmax(
            lattice_z_sample,
            dummy_Y,
            self.scaler_x_lattice,
            self.scaler_y_lattice,
        )
        return lattice_z_sample
    
    def _get_coord_generation(self, coord_z_sample_first: torch.Tensor):
        """
        Decode coordinate latent → coordinate features in original scale.
        """
        coord_z_sample = self.model_coord.decoder.forward(
            coord_z_sample_first.to(self.device).to(torch.float64)
        )
        # coord_z_sample is already a tensor on the right device
        coord_z_sample = coord_z_sample.detach().cpu().numpy()

        dummy_Y = self.Y_normed_coord[: coord_z_sample.shape[0]]
        coord_z_sample, _ = inv_minmax(
            coord_z_sample, dummy_Y, self.scaler_x_coord, self.scaler_y_coord
        )
        return coord_z_sample



    def _get_element_generation(self, element_z_sample_first: torch.Tensor):
        """
        Decode element latent → element+occupancy features in original scale.
        """
        with torch.no_grad():
            element_z_gen = self.model_element.decoder.forward(
                element_z_sample_first.to(self.device).to(torch.float64)
            )
        element_z_gen = element_z_gen.detach().cpu().numpy()

        # Build a container with correct shape then invert scaling
        # Shape like in original code: (batch, channels, features)
        # channels == 4, features == 143 for element+occupancy block
        batch_size = element_z_gen.shape[0]
        # For inverse scaling, we need a dummy X with same shape as training X_elem_occ
        n_channels = 4
        n_features = 143
        X_test_gen_array = np.zeros((batch_size, n_channels, n_features))
        X_test_gen_array[:, :4, 0:143] = element_z_gen[:, :4, 0:143]

        dummy_Y = self.Y_normed_elemocc[:batch_size]
        X_inv, _ = inv_minmax(
            X_test_gen_array,
            dummy_Y,
            self.scaler_x_elemocc,
            self.scaler_y_elemocc,
        )
        return X_inv  # shape: (batch, features, channels) after inv_minmax

    def _decode_latent(self, z: np.ndarray, step_index: int) -> Tuple[np.ndarray, float, Dict]:
        """
        Decode latent vector z into structure (FTCP), compute "measurement"
        observation and scalar reward.

        Returns
        -------
        obs : np.ndarray
            1-D observation array (measurement plot).
        reward : float
            Scalar reward.
        info : dict
            Extra info (z, ftcp, formation_energy, CIF path, etc.).
        """
        if z.ndim == 1:
            z = z[None, :]  # (1, D)

        latent_space = torch.tensor(z, dtype=torch.float64)
        # clip to combined_z_size in case someone passes longer z
        latent_space = latent_space[:, : self.combined_z_size]

        with torch.no_grad():
            parts_z = self.model_combined.decoder.forward(latent_space.to(self.device))
            element_z_rs, lattice_z_rs, coord_z_rs = self._rescale_and_divide(parts_z)

            element_z = self._get_element_generation(element_z_rs)
            lattice_z = self._get_lattice_generation(lattice_z_rs, )
            coord_z = self._get_coord_generation(coord_z_rs)

            # Build full FTCP representation template like in repo
            generation_template = self.X_array_origin[:1, :, :]
            generation_template = torch.tensor(generation_template).to(self.device)

            # element + occupancy
            generation_template[:1, 0:103, :4] = torch.tensor(element_z)[:1, 0:103, :4]
            generation_template[:1, 145:185, :4] = torch.tensor(element_z)[
                :1, 103:143, :4
            ]
            # lattice abc/angles
            x = torch.tensor(lattice_z[:, 0, :3])
            generation_template[:1, 103, :3] = x
            generation_template[:1, 104, :3] = torch.tensor(lattice_z)[:, 1, :3]
            # coordinates
            generation_template[:1, 105:145, :4] = torch.tensor(coord_z)[
                :1, :40, :
            ]

            ftcp = generation_template[:1, :185, :4].detach().cpu().numpy()

        # Optional: convert to CIF and write to disk (expensive)
        cif_path = None
        if self.save_cif:
            cif_folder = os.path.join(
                self.exp_name,
                self.exp_name2,
                f"ep_{self._episode_id:06d}",
                f"step_{step_index:03d}"
            )
            os.makedirs(cif_folder, exist_ok=True)
            (
                pred_formula,
                pred_abc,
                pred_ang,
                pred_latt,
                pred_site_coor,
                generated_elm_dict,
            ) = convert_cif(
                ftcp,
                max_elms=self.max_elms,
                max_sites=self.max_sites,
                elm_str=self.elm_str,
                to_CIF=True,
                folder_name=cif_folder,
                convertibility_only=False,
                print_error=False,
            )
            cif_path = os.path.join(cif_folder, "0.cif")

        # Default observation: flatten FTCP slice → "measurement plot"
        default_obs = ftcp.reshape(-1).astype(np.float32)



        # Default reward from XGBoost FE predictor
        formation_energy = None
        default_reward = 0.0
        if self.xgb_model is not None:
            fe_pred = self.xgb_model.predict(ftcp.reshape(1, -1))
            formation_energy = float(fe_pred[0])
            # RL maximizes reward, so we use -FE (more negative FE → higher reward)
            default_reward = -formation_energy

        info = {
            "z": z.squeeze().copy(),
            "ftcp": ftcp,
            "formation_energy": formation_energy,
            "cif_path": cif_path,
            "Formula": pred_formula,
        }

        ##  Also create a POSCAR file
        structure = Structure.from_file(cif_path)
        structure.to(filename=os.path.join(cif_folder, "POSCAR"))
        # structure = Structure.from_file(cif_folder+ '/'+ str(step_index) +'.cif')
        # structure.to(filename=cif_folder+'/POSCAR')

        # If user provided custom measurement_fn, override obs & reward
        if self.measurement_fn is not None:
            obs, reward = self.measurement_fn(ftcp, info)
        else:
            obs, reward = default_obs, default_reward

        return obs, float(reward), info

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # if seed is not None:
        #     np.random.seed(seed)

        # Sample z uniformly in [-bound, bound]
        self._episode_id += 1   ##added
        self._step_count = 0  ## added
        self._z = np.random.uniform(
            low=-self.bound,
            high=self.bound,
            size=(self.combined_z_size,),
        ).astype(np.float64)
        #self._step_count = 0

        obs, _, info = self._decode_latent(self._z, step_index=0)
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"Invalid action {action}, expected 0..{self.action_space.n - 1}"

        dim = action // 2
        sign = +1.0 if (action % 2 == 0) else -1.0

        # Update latent state
        self._z[dim] += sign * self.step_size
        # Clip to bounds
        self._z = np.clip(self._z, -self.bound, self.bound)

        self._step_count += 1
        obs, formation_E_reward, info = self._decode_latent(self._z, step_index=self._step_count)
        

        print(f'Step: {self._step_count}\n---------------------------\n {info["Formula"]}\n -------------')
        ######### Rewards############
        ### 1. Adsorption energy from fairchem MD
        # E_ads= adsorption_energy_h_from_poscar(poscar_path=info['cif_path']+'/POSCAR', layers = 5, miller=(1,0,0),vacuum=10, site_atom_index=0, height=2, fmax=0.01,fix_n_bottom_layers=0,reference='H2',return_details= True)[0]
        # Path to the Python executable in your target environment
        TARGET_PYTHON = "/home/anupam/softwares/miniconda3/envs/fairchem/bin/python"
        #  Path to the script you want to run
        ADS_SCRIPT = "/home/anupam/Anupam/model_based_RL/generate_material_space_CDVAE_variations/Crystal_LS_RL/MD_eSen.py"
        ## input POSCAR file
        path = info['cif_path'].rsplit('/',1)[0]
        #path= '../../csllm_github'
        print(f'Path of structure file at this step: {path}')
        try:
            # Run the script using the target interpreter
            result = subprocess.run(
                [TARGET_PYTHON, ADS_SCRIPT, path+'/POSCAR'],
                capture_output=True,
                text=True,
                check=True  # Raise an error if the script fails
            )
            print(f"Farichem E_ads Output: {float(result.stdout)} eV")
            fairchem_E_ads = float(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Fairchem Failed: {e}")
            print("Stderr:\n", e.stderr)
        
        ### 2. Predict i0 for HER from E_ads   # ## read-out catalytic activity (i0) using a volcano plot
        ### Norskov et. al    log (i0) = log10(i0_max) = -16.9 |Del G | 
        Free_E = fairchem_E_ads + 0.24   # eV    
        i0_max = 10  ## A/m2
        kB = 8.617333262e-5  # eV/K
        T= 298
        x = Free_E / (kB * T)
        # 3. Volcano (Nørskov kinetic model)
        if Free_E <= 0:   # strong-binding side
            i0 = i0_max / (1 + np.exp(-x))
        else:             # weak-binding side
            i0 = i0_max * np.exp(-x) / (1 + np.exp(-x))
        print(f'Predicted i0 from volcano plot: {i0} A/m2')

        ### 3. Create a synthetic CV loop from from the predicted i0
        ## Call the CV creation tool 
        generate_and_save(
                        num_train=0,
                        num_val=1,
                        seed= 45,
                        image_size=128,
                        Emax=1.0,
                        Emin=-0.3,
                        Imax=60e-3,
                        Imin=-60e-3,
                        save_imgs_dir=path,
                        pred_target_dir=path,
                        HER_i0=i0,
                    )
        ### 4. Get the overpotential for given (60e-3 mA) using VAE
        Overpot, HERi0 = infer_VAE([
                            "--checkpoint", "working_VAE_CV/outputs/vae_epoch_040.pt",
                            "--image", path+"/00000.png",
                            "--out_dir", path,
                            "--gt_i0_is_log10",
                        ])
        print(f'Predicted Overpotential:{float(Overpot)}, i0 for HER : {float(HERi0)}')
        ### 5. Predict reward from overpotential 
        Reward_overpot = 1/abs(float(Overpot))

        ### 6. CSLLM synthesizability reward
        ## call CSLLM in a different env
        Reward_csllm = 0
        # TARGET_PYTHON = "/home/anupam/softwares/miniconda3/envs/csllm_gpu/bin/pythonn"
        # ADS_SCRIPT = "/home/anupam/Anupam/model_based_RL/csllm_github/CSLLM-main/csllm_peft.py"
        # try:
        #     # Run the script using the target interpreter
        #     csllm_result = subprocess.run(
        #             [TARGET_PYTHON, ADS_SCRIPT, '--poscar_path',  path+'/POSCAR', 
        #             '--base_mode', '/home/anupam/Anupam/model_based_RL/csllm_github/csllm_model/llama3-8bf-hf',
        #             '--synthesis_adapter',  '/home/anupam/Anupam/model_based_RL/csllm_github/csllm_model/synthesis_llm_llama3',
        #             '--method_adapter', '/home/anupam/Anupam/model_based_RL/csllm_github/csllm_model/method_llm_llama3',
        #             '--precursor_adapter', '/home/anupam/Anupam/model_based_RL/csllm_github/csllm_model/precursor_llm_llama3' ],
        #         capture_output=True,
        #         text=True,
        #         check=True  # Raise an error if the script fails
        #     )
        #     Reward_csllm = csllm_result[0]  ### Need to check this
        # except subprocess.CalledProcessError as e:
        #     print(f"csllm Failed: {e}")
        #     print("Stderr:\n", e.stderr)

        ### 9. Get total reward
        Reward =  Reward_overpot - 2 + Reward_csllm  + formation_E_reward
        print(f'Total reward at this step : {Reward}')

        done = self._step_count >= self.max_steps
        truncated = False  # you can set a separate truncation condition if desired

        return obs, Reward, done, truncated, info

    def render(self):
        # You can add a custom visualization here (e.g. print z, FE, etc.)
        if self._z is not None:
            print(f"Step {self._step_count}, z[0:3]={self._z[:3]} ...")

    def close(self):
        pass

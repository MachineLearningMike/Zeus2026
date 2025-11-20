import os
import zarr
import numpy as np
from threading import Lock
from zeus_dataset_codec import encode, decode
import numcodecs


class ERA5ZarrShards:
    """
    ERA5 dataset stored as Zarr shards.
    Dynamic shard creation + absolute indexing.
    """

    def __init__(self, base_dirs, shard_shapes, max_slices=None, chunklen=1, auto_expand=False):
        """
        base_dir: parent directory where dynamic shard_XXXXX.zarr directories will be created
        shard_shapes: dict(var -> (H, W))
        max_slices: required if auto_expand=False
        auto_expand: allow resize
        """
        self.base_dirs = base_dirs
        self.shard_shapes = shard_shapes
        self.chunklen = chunklen
        self.auto_expand = auto_expand
        self.max_slices = max_slices if not auto_expand else None

        if not auto_expand and max_slices is None:
            raise ValueError("max_slices must be provided when auto_expand=False")

        [os.makedirs(base_dir, exist_ok=True) for base_dir in self.base_dirs]

        # dynamic structures
        self.shards = {}       # shard_idx -> zarr group
        self.locks = {}        # shard_idx -> Lock object
        self.meta_codec = numcodecs.MsgPack()


    # ----------------------------------------------------------------------
    # INTERNAL: ensure shard exists
    # ----------------------------------------------------------------------
    def _get_or_create_shard(self, shard_idx):
        if shard_idx in self.shards:
            return self.shards[shard_idx]

        # Create new shard path
        base_dir = self.base_dirs[shard_idx % len(self.base_dirs)]
        shard_path = os.path.join(base_dir, f"shard_{shard_idx:05d}.zarr")
        os.makedirs(shard_path, exist_ok=True)

        # Open the root zarr group
        root = zarr.open(shard_path, mode='a')

        # Create missing datasets
        for var, (H, W) in self.shard_shapes.items():

            if var not in root:
                if self.auto_expand:
                    root.create_dataset(
                        var,
                        shape=(0, H, W),
                        maxshape=(None, H, W),
                        chunks=(self.chunklen, H, W),
                        dtype=np.uint16,
                    )
                else:
                    root.create_dataset(
                        var,
                        shape=(self.max_slices, H, W),
                        chunks=(self.chunklen, H, W),
                        dtype=np.uint16,
                    )

            if var + "_meta" not in root:
                if self.auto_expand:
                    root.create_dataset(
                        var + "_meta",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=object,
                        object_codec=self.meta_codec,
                        chunks=(self.chunklen,),
                    )
                else:
                    root.create_dataset(
                        var + "_meta",
                        shape=(self.max_slices,),
                        dtype=object,
                        object_codec=self.meta_codec,
                        chunks=(self.chunklen,),
                    )

        # Register the shard + lock
        self.shards[shard_idx] = root
        self.locks[shard_idx] = Lock()
        return root


    # ----------------------------------------------------------------------
    # PUBLIC: Write absolute-index slice
    # ----------------------------------------------------------------------
    def add_slice(self, shard_idx, slice_idx, slice_dict):
        root = self._get_or_create_shard(shard_idx)

        encoded, meta = encode(slice_dict)

        with self.locks[shard_idx]:
            for var, arr in encoded.items():
                arr = np.squeeze(arr)
                expected = self.shard_shapes[var]

                if arr.shape != expected:
                    raise ValueError(f"{var}: got {arr.shape}, expected {expected}")

                # Expand if needed
                if self.auto_expand:
                    current_len = root[var].shape[0]
                    if slice_idx >= current_len:
                        root[var].resize((slice_idx + 1, *arr.shape))
                        root[var + "_meta"].resize((slice_idx + 1,))
                else:
                    if slice_idx >= self.max_slices:
                        raise ValueError(
                            f"slice_idx {slice_idx} out of range (max={self.max_slices - 1})"
                        )

                root[var][slice_idx] = arr
                root[var + "_meta"][slice_idx] = meta[var]


    # ----------------------------------------------------------------------
    def slice_exists(self, shard_idx, slice_idx):
        try:
            root = self.shards[shard_idx]
            for var in self.shard_shapes:
                assert root[var][slice_idx] is not None
                assert root[var + "_meta"][slice_idx] is not None
            return True
        except:
            return False
    
    
    def read_slice(self, shard_idx, slice_idx):
        root = self._get_or_create_shard(shard_idx)

        encoded_slice = {}
        meta_slice = {}

        for var in self.shard_shapes:
            encoded_slice[var] = root[var][slice_idx]
            meta_slice[var] = root[var + "_meta"][slice_idx]

        return decode(encoded_slice, meta_slice)


    # ----------------------------------------------------------------------
    def get_batch(self, shard_idx, slice_indices):
        batch = [self.read_slice(shard_idx, i) for i in slice_indices]
        return {
            var: np.stack([s[var] for s in batch], axis=0)
            for var in self.shard_shapes
        }



# ----------------------- Demo -----------------------
if __name__ == "__main__":
    shared_shape = {
        "2m_temperature": (721, 1440),
        "surface_pressure": (721, 1440),
        "u10": (721, 1440),
        "precipitation": (721, 1440),
    }

    dataset = ERA5ZarrShards(
        base_dirs=["data_safe"],
        shard_shapes=shared_shape,
        max_slices=1000,
        chunklen=1,
        auto_expand=True
    )

    slice_dict = {
        "2m_temperature": np.ones((721,1440), dtype=np.float32),
        "surface_pressure": np.random.rand(721,1440).astype(np.float32),
        "u10": np.random.rand(721,1440).astype(np.float32),
        "precipitation": np.random.rand(721,1440).astype(np.float32),
    }
    print(slice_dict["surface_pressure"])

    dataset.add_slice(0, 200, slice_dict)
    decoded = dataset.read_slice(0, 200)
    print(decoded["surface_pressure"])
    
    print("Decoded matches original:",
          np.allclose(decoded["surface_pressure"], slice_dict["surface_pressure"]))
import os
import zarr
import numpy as np
from threading import Lock
from zeus_dataset_codec import encode, decode
import numcodecs


class ERA5ZarrShards:
    """
    ERA5 dataset stored as Zarr shards.
    Uses absolute indexing: you choose slice_idx explicitly.
    """

    def __init__(self, shard_paths, shard_shapes, max_slices=None, chunklen=1, auto_expand=False):
        """
        shard_paths : list of zarr directories, one per shard
        shard_shapes : dict(var -> (H, W))
        max_slices : int or None. Required if auto_expand=False.
        auto_expand : if True, dataset grows dynamically with resize().
        """
        self.shard_paths = shard_paths
        self.shard_shapes = shard_shapes
        self.chunklen = chunklen
        self.auto_expand = auto_expand
        self.max_slices = max_slices if not auto_expand else None

        if not auto_expand and max_slices is None:
            raise ValueError("max_slices must be provided if auto_expand=False")

        self.locks = [Lock() for _ in shard_paths]
        self.shards = []
        self.meta_codec = numcodecs.MsgPack()  # store dicts safely

        for path in shard_paths:
            os.makedirs(path, exist_ok=True)
            root = zarr.open(path, mode='a')

            # Create datasets if missing
            for var, (H, W) in shard_shapes.items():

                if var not in root:
                    if auto_expand:
                        # Start empty but expandable
                        root.create_dataset(
                            var,
                            shape=(0, H, W),
                            maxshape=(None, H, W),
                            chunks=(chunklen, H, W),
                            dtype=np.uint16,
                        )
                    else:
                        # Preallocate fixed space
                        root.create_dataset(
                            var,
                            shape=(max_slices, H, W),
                            chunks=(chunklen, H, W),
                            dtype=np.uint16,
                        )

                # Metadata dataset
                if var + "_meta" not in root:
                    if auto_expand:
                        root.create_dataset(
                            var + "_meta",
                            shape=(0,),
                            maxshape=(None,),
                            dtype=object,
                            object_codec=self.meta_codec,
                            chunks=(chunklen,),
                        )
                    else:
                        root.create_dataset(
                            var + "_meta",
                            shape=(max_slices,),
                            dtype=object,
                            object_codec=self.meta_codec,
                            chunks=(chunklen,),
                        )

            self.shards.append(root)

    # ----------------------------------------------------------------------

    def add_slice(self, shard_idx, slice_idx, slice_dict):
        """
        Write a slice at an absolute position slice_idx.
        encode/decode untouched.
        """
        root = self.shards[shard_idx]

        encoded, meta = encode(slice_dict)

        with self.locks[shard_idx]:

            for var, arr in encoded.items():
                arr = np.squeeze(arr)
                expected = self.shard_shapes[var]

                if arr.shape != expected:
                    raise ValueError(f"{var}: got {arr.shape}, expected {expected}")

                # Auto-expansion logic
                if self.auto_expand:
                    current_len = root[var].shape[0]
                    if slice_idx >= current_len:
                        new_len = slice_idx + 1
                        root[var].resize((new_len, *root[var].shape[1:]))
                        root[var + "_meta"].resize((new_len,))
                else:
                    # Fixed size shard
                    if slice_idx >= self.max_slices:
                        raise ValueError(
                            f"slice_idx={slice_idx} out of range (max={self.max_slices - 1})"
                        )

                # Write encoded data + metadata
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
        root = self.shards[shard_idx]

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
            var: np.stack([b[var] for b in batch], axis=0)
            for var in self.shard_shapes
        }


# ----------------------- Demo -----------------------
if __name__ == "__main__":
    shard_paths = ["data_safe/2015-01.zarr", "data_safe/2015-02.zarr"]
    shared_shape = {
        "2m_temperature": (721, 1440),
        "surface_pressure": (721, 1440),
        "u10": (721, 1440),
        "precipitation": (721, 1440),
    }

    dataset = ERA5ZarrShards(shard_paths, shared_shape, max_slices=1000, chunklen=1, auto_expand=True)

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
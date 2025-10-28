import numpy as np
import json
import os

# ---------- Core scaled uint16 encoder ----------
def encode_uint16(arr, offset=None, scale=None, clip=True):
    arr = np.asarray(arr, dtype=np.float32)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)

    if offset is None:
        offset = vmin
    if scale is None:
        scale = (vmax - vmin) / 65535.0 if vmax != vmin else 1.0

    norm = (arr - offset) / scale
    if clip:
        norm = np.clip(norm, 0, 65535)
    else:
        if np.any(norm < 0) or np.any(norm > 65535):
            raise ValueError("Data outside representable 16-bit range")

    return np.round(norm).astype(np.uint16), {
        "type": "uint16",
        "offset": float(offset),
        "scale": float(scale),
        "min": float(vmin),
        "max": float(vmax),
    }

def decode_uint16(arr16, meta):
    return arr16.astype(np.float32) * meta["scale"] + meta["offset"]


# ---------- float16 encoder ----------
def encode_float16(arr):
    return arr.astype(np.float16), {"type": "float16"}

def decode_float16(arr16, _meta):
    return arr16.astype(np.float32)


# ---------- Hybrid dtype selection ----------
# Predefined critical variables that need high precision
CRITICAL_VARS = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind', "2m_dewpoint_temperature", "surface_pressure"]

def choose_dtype(var_name, arr):
    if var_name in CRITICAL_VARS:
        return "uint16"
    else:
        # Moderate range -> float16
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if vmax - vmin > 1e4:  # very large range? could use uint16
            return "uint16"
        return "float16"


# ---------- Multi-variable encoding ----------
def encode_variables(var_dict):
    encoded, metadata = {}, {}

    for name, arr in var_dict.items():
        dtype = choose_dtype(name, arr)
        if dtype == "uint16":
            enc, meta = encode_uint16(arr)
        elif dtype == "float16":
            enc, meta = encode_float16(arr)
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
        encoded[name] = enc
        metadata[name] = meta
        # print(f"[encode16_hybrid] {name}: {dtype} (min={arr.min():.2f}, max={arr.max():.2f})")

    return encoded, metadata

def save_variables(filename, encoded_dict, metadata_dict):
    save_dict = {f"data_{k}": v for k, v in encoded_dict.items()}
    save_dict["meta"] = json.dumps(metadata_dict)
    np.savez_compressed(filename, **save_dict)
    return

def load_variables(filename):
    with np.load(filename, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        encoded = {k.replace("data_", ""): npz[k] for k in npz.files if k.startswith("data_")}
    return encoded, meta

def load_decoded_variables(filename):
    encoded, meta = load_variables(filename)
    decoded = {}
    for k, arr in encoded.items():
        t = meta[k]["type"]
        if t == "uint16":
            decoded[k] = decode_uint16(arr, meta[k])
        elif t == "float16":
            decoded[k] = decode_float16(arr, meta[k])
    return decoded


#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    import numpy as np
    # from this_module import encode_variables, save_variables, load_decoded_variables

    eds_data = {
        "2m_temperature": np.random.uniform(200, 320, (24, 721, 1440)).astype(np.float32),
        "surface_pressure": np.random.uniform(87000, 108000, (24, 721, 1440)).astype(np.float32),
        "u10": np.random.uniform(-30, 30, (24, 721, 1440)).astype(np.float32),
        "precipitation": np.random.uniform(0.0, 1.0, (24, 721, 1440)).astype(np.float32),
    }

    encoded, meta = encode_variables(eds_data)
    save_variables("eds_hybrid.npz", encoded, meta)

    decoded = load_decoded_variables("eds_hybrid.npz")

    for k in eds_data:
        diff = np.max(np.abs(decoded[k] - eds_data[k]))
        print(f"{k}: max abs diff = {diff:.10f}, shape = {decoded[k].shape}, dtype = {decoded[k].dtype}")


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
def encode(var_dict):
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

def decode(encoded, meta):
    decoded = {}
    for k, arr in encoded.items():
        t = meta[k]["type"]
        if t == "uint16":
            decoded[k] = decode_uint16(arr, meta[k])
        elif t == "float16":
            decoded[k] = decode_float16(arr, meta[k])
    return decoded    

def save_variables(path, encoded_dict, metadata_dict):
    save_dict = {f"data_{k}": v for k, v in encoded_dict.items()}
    save_dict["meta"] = json.dumps(metadata_dict)
    np.savez_compressed(path, **save_dict)
    return

def load_variables(path):
    with np.load(path, allow_pickle=False) as npz:
        meta = json.loads(str(npz["meta"]))
        encoded = {k.replace("data_", ""): npz[k] for k in npz.files if k.startswith("data_")}
    return encoded, meta

def save(path, array_dict):
    encoded, meta = encode(array_dict)
    save_variables(path, encoded, meta)

def load(path):
    encoded, meta = load_variables(path)
    array_dict = decode(encoded, meta)
    return array_dict

#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    import numpy as np
    # from this_module import encode, save_variables, load_decoded_variables

    array_dict = {
        "2m_temperature": np.random.uniform(200, 320, (24, 721, 1440)).astype(np.float32),
        "surface_pressure": np.random.uniform(87000, 108000, (24, 721, 1440)).astype(np.float32),
        "u10": np.random.uniform(-30, 30, (24, 721, 1440)).astype(np.float32),
        "precipitation": np.random.uniform(0.0, 1.0, (24, 721, 1440)).astype(np.float32),
    }
    
    array_dict = {
        "2m_temperature": np.random.uniform(200, 320, (721, 1440)).astype(np.float32),
        "surface_pressure": np.random.uniform(87000, 108000, (721, 1440)).astype(np.float32),
        "u10": np.random.uniform(-30, 30, (721, 1440)).astype(np.float32),
        "precipitation": np.random.uniform(0.0, 1.0, (721, 1440)).astype(np.float32),
    }
    
    # print(array_dict["u10"])

    path = "d:\downloads\cds_hybrid.npz"

    print("saving...")
    save(path, array_dict)
    
    print("loading...")
    array_dict_restored = load(path)

    for k in array_dict:
        rel_diff = np.max(np.abs(array_dict_restored[k] - array_dict[k])) / np.mean(np.abs(array_dict_restored[k]))
        print(f"====== {k}: max rel diff = {rel_diff:.10f}, mean: {np.mean(np.abs(array_dict_restored[k]))}, shape = {array_dict[k].shape}, dtype = {array_dict_restored[k].dtype}")

    # print(array_dict_restored["u10"])

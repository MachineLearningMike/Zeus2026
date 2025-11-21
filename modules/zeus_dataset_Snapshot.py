
import torch
from datetime import datetime, timezone

from zeus_dateset_orbital_angle import earth_orbit_angle

class Snapshot:
    # temporary. can be added and changed any time
    meteo_dims = 85
    geo_dims = 5        # lat, lon, orograph, land_sea_mask, soil-type
    tempo_dims = 2      # orbit: sign_earth_orbital_angle, cascade: cascade_embedding_placeholder
    height = 721
    width = 1440
    
    def __init__(self, year, month, day, hour, meteo = None, shared_geo = None, tempo = None, device='cpu'):
        assert year is not None and month is not None and day is not None and hour is not None
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        
        self._device = device
        self._fresh = True

        if meteo is not None:
            assert torch.is_tensor(meteo)
            assert meteo.shape == (self.meteo_dims, self.height, self.width)            
        if shared_geo is not None:
            assert torch.is_tensor(shared_geo)
            assert shared_geo.shape == (self.geo_dims, self.height, self.width)            
        if tempo is not None:
            assert torch.is_tensor(tempo)
            assert tempo.shape == (self.tempo_dims, self.height, self.width)            

        self._meteo_tensor = meteo
        self._geo_tensor = shared_geo
        self._tempo_tensor = tempo
        
    #-------------------- device -----------------
    def _get_device(self):
        devices = [
            self._meteo_tensor.device.device if self._meteo_tensor is not None else None,
            self._geo_tensor.device.device if self._geo_tensor is not None else None,
            self._tempo_tensor.device.device if self._tempo_tensor is not None else None,
        ]
        true_devices = [d for d in devices if d is not None]
        if len(true_devices) <= 0:
            device = None
        else:
            cpu_devices = [d for d in devices if d == "cpu"]
            if len(cpu_devices) == 0:
                device = "cuda"
            elif len(cpu_devices) == len(true_devices):
                device = "cpu"
            else: device = None # mixed devices
        return device
    def _set_device(self, device):
        tensors = [self._meteo_tensor, self._geo_tensor, self._tempo_tensor]
        true_tensors = [arr for arr in tensors if arr is not None]
        [arr.to(device) for arr in true_tensors]
        return
    def to(self, device):
        self._device = device
    device = property(
        fget = _get_device,
        fset = _set_device,
    )

    #------------------- fresh ---------------------
    def _get_fresh(self):
        return self._fresh
    def _set_fresh(self, fresh):
        self._fresh = fresh
    fresh = property(
        fget = _get_fresh,
        fset = _set_fresh,
    )
    
    #------------------ meteo -----------------------
    def _get_meteo_tensor(self):
        return self._meteo_tensor
    def _set_meteo_tensor(self, tensor):
        self._meteo_tensor = tensor
    meteo = property(
        fget = _get_meteo_tensor,
        fset = _set_meteo_tensor,
    )
    
    #------------------ geo -----------------------
    def _get_geo_tensor(self):
        return self._geo_tensor
    def _set_geo_tensor(self, tensor):
        self._geo_tensor = tensor
    geo = property(
        fget = _get_geo_tensor,
        fset = _set_geo_tensor,
    )
    
    #------------------ tempo -----------------------
    def _get_tempo_tensor(self):
        return self._tempo_tensor
    def _set_tempo_tensor(self, tensor):
        self._tempo_tensor = tensor
    def create_tempo(self):
        dt = datetime(self.year, self.month, self.day, self.hour)
        orbatal_angle = earth_orbit_angle(dt)
        # self._tempo_tensor[1] is a placeholder for cascade embedding, which is part of model parameters.
        self._tempo_tensor = torch.ones(self.tempo_dims, self.height, self.width) * torch.sin(torch.deg2rad(orbatal_angle))
        
    tempo = property(
        fget = _get_tempo_tensor,
        fset = _set_tempo_tensor,
    )
    
    #------------------ input tensor  --------------------
    def _get_input_tensor(self):
        if self._meteo_tensor is None or self._geo_tensor is None or self._tempo_tensor is None: return None
        tensor = torch.concat([self._meteo_tensor, self._geo_tensor, self._tempo_tensor], dim=0)
        return tensor
    def _set_input_tensor(self, tensor):
        assert tensor.shape[0] == self.meteo_dims + self.geo_dims + self.tempo_dims
        self._meteo_tensor = tensor[:self.meteo_dims]
        self._geo_tensor = tensor[self.meteo_dims : self.meteo_dims + self.geo_dims]
        self._tempo_tensor = tensor[self.meteo_dims + self.geo_dims :]
        return
    input_tensor = property(
        fget = _get_input_tensor,
        fset = _set_input_tensor,
    )

    #------------------ output tensor  --------------------
    def _get_output_tensor(self):
        return self._meteo_tensor
    def _set_output_tensor(self, tensor):
        assert tensor.shape[0] == self.meteo_dims
        self._meteo_tensor = tensor
        return
    input_tensor = property(
        fget = _get_output_tensor,
        fset = _set_output_tensor,
    )

    #------------------ miscellaneout --------------------
    def delete(self):
        self._device = 'cpu'
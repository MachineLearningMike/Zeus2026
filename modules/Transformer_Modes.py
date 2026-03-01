from enum import Enum

class AttnConn(Enum):
    Nothing = 0
    Global = 1
    Axial = 2
    Shift = 3
    
class PEConn(Enum):
    Nothing = 0
    Global = 1
    Axial = 2
    
class PEType(Enum):
    Nothing = 'nothing'
    Learnable = 'learnable'
    Sinusoidal = 'sinusoidal'
    Linear = 'linear'
    
class Flash(Enum):
    No = False
    Yes = True
    
class PBias(Enum):
    Nothing = 0
    Learnable = 1
    
class TransForm:
    def __init__(self, 
                attnconn=AttnConn.Global,
                peconn=PEConn.Nothing,
                petype=PEType.Nothing,
                flash=Flash.No,
                pbias=PBias.Nothing,
                ) -> None:
        # assert self.is_valid(attnconn, peconn, petype, flash, pbias), "Invalid Transformer mode"
        
        self.attnconn = attnconn
        self.peconn = peconn
        self.petype = petype
        self.flash = flash
        self.pbias = pbias
    
    @staticmethod
    def is_valid(conn, flash, penc, pshare, pbias):
        """Check that no invalid pair of modes is active together."""
        
        invalid_modes = [
            (Flash.Yes, AttnConn.Shift),        # Can't use flash with shift connection
            (Flash.Yes, PBias.Learnable),       # Can't use flash with learnable bias
        ]
        
        for im in invalid_modes:
            # Count how many parameters match this invalid tuple
            count = sum([
                conn in im,
                flash in im, 
                penc in im, 
                pshare in im, 
                pbias in im
            ])
            
            # If 2 or more match, it's invalid (both conditions in the tuple are true)
            if count >= 2:
                return False
        
        return True

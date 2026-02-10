from enum import Enum

class Conn(Enum):
    Nothing = 0
    Global = 1
    Axial = 3
    Shift = 4
    
class Flash(Enum):
    Nothing = 0
    UseFlash = 1
    NoFlash = 2
    
class PEnc(Enum):
    Nothing = 0
    Learnable = 1
    Sinusodial = 2
    Linear = 3

class PShare(Enum):
    Global = 1
    Axial = 2
    
class PBias(Enum):
    Nothing = 0
    Learnable = 1
    
class TModes:
    def __init__(self, conn=Conn.Nothing, 
                 flash=Flash.Nothing, 
                 penc=PEnc.Nothing, 
                 pshare=PShare.Nothing, 
                 pbias=PBias.Nothing) -> None:
        
        # Validate before assignment
        if not self.is_valid(conn, flash, penc, pshare, pbias):
            raise ValueError("Invalid mode combination")
        
        self.conn = conn
        self.flash = flash
        self.penc = penc
        self.pshare = pshare
        self.pbias = pbias
    
    @staticmethod
    def is_valid(conn, flash, penc, pshare, pbias):
        """Check that no invalid pair of modes is active together."""
        
        invalid_modes = [
            (Flash.UseFlash, Conn.Shift),        # Can't use flash with shift connection
            (Flash.UseFlash, PBias.Learnable)    # Can't use flash with learnable bias
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

# Test it
if __name__ == "__main__":
    # This should fail: Both Flash.UseFlash and Conn.Shift
    try:
        t1 = TModes(conn=Conn.Shift, flash=Flash.UseFlash)
        print("ERROR: Should have failed!")
    except ValueError as e:
        print("✓ Correctly rejected Flash.UseFlash + Conn.Shift")
    
    # This should work: Only Flash.UseFlash
    t2 = TModes(flash=Flash.UseFlash)
    print("✓ Accepted Flash.UseFlash alone")
    
    # This should work: Only Conn.Shift  
    t3 = TModes(conn=Conn.Shift)
    print("✓ Accepted Conn.Shift alone")
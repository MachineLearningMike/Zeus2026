
def halve_window_size_optimized_flexibly(sWin, nWin, pWin):
    """
    Optimized version using mathematical approach.
    new_sWin is the smallest number >= sWin/2 that meets constraints.
    """
    new_sWin = []
    for i in range(2):
        # Start from ceil(sWin[i]/2) - smallest number not smaller than sWin/2
        min_size = (sWin[i] + 1) // 2
        
        # Step 1: Find smallest size >= min_size divisible by nWin[i]
        remainder = min_size % nWin[i]
        size = min_size if remainder == 0 else min_size + (nWin[i] - remainder)
        
        # Step 2: Ensure window_size is divisible by pWin
        window_size = size // nWin[i]
        remainder_pWin = window_size % pWin
        if remainder_pWin != 0:
            # Increase window_size to next multiple of pWin
            window_size += (pWin - remainder_pWin)
            # Recalculate size
            size = window_size * nWin[i]
        
        new_sWin.append(size)
    
    return tuple(new_sWin)


if __name__ == "__main__":

    # Example usage
    sWin = (45, 90)
    nWin = (1, 1)
    pWin = 1

    result = halve_window_size_optimized_flexibly(sWin, nWin, pWin)
    print(f"Input: sWin={sWin}, nWin={nWin}, pWin={pWin}")
    print(f"Min size required: {[s//2 + 1 for s in sWin]}")
    print(f"New window size: {result}")

    # Verify
    for i in range(2):
        print(f"\nDimension {i}:")
        print(f"  min required: {sWin[i]//2 + 1}")
        print(f"  new_sWin={result[i]}")
        print(f"  greater than sWin/2: {result[i] > sWin[i]/2}")
        print(f"  divisible by nWin[{i}]={nWin[i]}: {result[i] % nWin[i] == 0}")
        print(f"  window_size={result[i] // nWin[i]}")
        print(f"  divisible by pWin={pWin}: {(result[i] // nWin[i]) % pWin == 0}")
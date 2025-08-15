from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import List, Union
from datetime import date

class DistributedDateStorage:
    def __init__(self, drive_paths: List[Union[str, Path]], max_workers: int = 4):
        self.drives = [Path(p) for p in drive_paths]
        for drive in self.drives:
            drive.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def __del__(self):
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)

    def _days_since_epoch(self, year: int, month: int, day: int) -> int:
        """Compute number of days since 1970-01-01."""
        d = date(year, month, day)
        epoch = date(1970, 1, 1)
        return (d - epoch).days

    def _drive_for_date(self, year: int, month: int, day: int) -> Path:
        days = self._days_since_epoch(year, month, day)
        idx = days % len(self.drives)
        return self.drives[idx]

    def _filename(self, year: int, month: int, day: int) -> str:
        """Generate filename YYYY-MM-DD.npy"""
        return f"{year:04d}-{month:02d}-{day:02d}.npy"

    def save(self, year: int, month: int, day: int, data: np.ndarray):
        drive = self._drive_for_date(year, month, day)
        filepath = drive / self._filename(year, month, day)
        np.save(filepath, data)

    def _load_single(self, year: int, month: int, day: int) -> np.ndarray:
        drive = self._drive_for_date(year, month, day)
        filepath = drive / self._filename(year, month, day)
        return np.load(filepath)

    def load_dates(self, dates: List[tuple[int, int, int]]) -> np.ndarray:
        """Load multiple dates in parallel, concatenated in date order."""
        # Sort by (year, month, day)
        dates_sorted = sorted(dates)

        results = {}
        future_map: dict[Future, tuple[int,int,int]] = {
            self.executor.submit(self._load_single, y, m, d): (y, m, d)
            for y, m, d in dates_sorted
        }

        for future in as_completed(future_map):
            key = future_map[future]
            results[key] = future.result()

        concatenated = np.concatenate([results[key] for key in dates_sorted])
        return concatenated

# -----------------------
# Example usage
if __name__ == "__main__":
    drives = [
        "/mnt/drive1/data",
        "/mnt/drive2/data",
        "/mnt/drive3/data"
    ]
    storage = DistributedDateStorage(drives, max_workers=6)

    # Save example
    storage.save(2025, 8, 10, np.random.rand(100, 5))
    storage.save(2025, 8, 11, np.random.rand(200, 5))
    storage.save(2025, 8, 12, np.random.rand(150, 5))

    # Load multiple dates
    dates_to_load = [(2025, 8, 10), (2025, 8, 12), (2025, 8, 11)]
    data = storage.load_dates(dates_to_load)
    print(data.shape)

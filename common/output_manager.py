"""
Output Manager for Micromouse RL Project
Provides standardized output directory management across all phases.
"""
import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class OutputManager:
    """Manages output directory structure for training experiments"""

    def __init__(self, phase_name: str):
        """
        Initialize output manager for a specific phase.

        Args:
            phase_name: Name of the phase (e.g., 'phase1_open', 'phase2_slalom', 'phase3_maze')
        """
        self.phase_name = phase_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup directories
        self.base_dir = Path("outputs") / phase_name
        self.archive_dir = self.base_dir / "archive" / self.timestamp
        self.latest_dir = self.base_dir / "latest"

        # Create archive directory
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, filename: str, use_latest: bool = False) -> Path:
        """
        Get full path for output file.

        Args:
            filename: Name of the output file
            use_latest: If True, return path in latest/ instead of archive/

        Returns:
            Full path to the output file
        """
        if use_latest:
            return self.latest_dir / filename
        return self.archive_dir / filename

    def save_metrics(self, metrics: Dict[str, Any], phase_specific: Optional[Dict[str, Any]] = None):
        """
        Save metrics in standardized JSON format.

        Args:
            metrics: Dictionary of common metrics
            phase_specific: Optional phase-specific metrics
        """
        output = {
            "timestamp": self.timestamp,
            "phase": self.phase_name,
            **metrics
        }

        if phase_specific:
            output["phase_specific"] = phase_specific

        metrics_path = self.archive_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Metrics saved to {metrics_path}")

    def save_model_info(self, info: Dict[str, Any]):
        """
        Save model information as text file.

        Args:
            info: Dictionary containing model information
        """
        info_path = self.archive_dir / "model_info.txt"

        with open(info_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Model Information - {self.phase_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("=" * 60 + "\n\n")

            for key, value in info.items():
                f.write(f"{key}: {value}\n")

        print(f"Model info saved to {info_path}")

    def update_latest(self):
        """
        Update latest/ directory with current archive contents.
        Removes old latest/ and copies current archive.
        """
        # Remove old latest if exists
        if self.latest_dir.exists():
            shutil.rmtree(self.latest_dir)

        # Copy archive to latest
        shutil.copytree(self.archive_dir, self.latest_dir)
        print(f"Latest directory updated: {self.latest_dir}")

    def finalize(self, summary: Optional[str] = None):
        """
        Finalize output by updating latest/ and optionally printing summary.

        Args:
            summary: Optional summary message to print
        """
        self.update_latest()

        print("\n" + "=" * 60)
        print(f"Output Management Complete")
        print("=" * 60)
        print(f"Archive: {self.archive_dir}")
        print(f"Latest:  {self.latest_dir}")

        if summary:
            print("\n" + summary)

        print("=" * 60)

    @staticmethod
    def load_latest_metrics(phase_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metrics from latest/ directory.

        Args:
            phase_name: Name of the phase

        Returns:
            Metrics dictionary or None if not found
        """
        metrics_path = Path("outputs") / phase_name / "latest" / "metrics.json"

        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            return json.load(f)

    @staticmethod
    def list_archives(phase_name: str) -> list:
        """
        List all archived experiments for a phase.

        Args:
            phase_name: Name of the phase

        Returns:
            List of archive directory names (timestamps)
        """
        archive_base = Path("outputs") / phase_name / "archive"

        if not archive_base.exists():
            return []

        return sorted([d.name for d in archive_base.iterdir() if d.is_dir()])

    @staticmethod
    def load_archive_metrics(phase_name: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Load metrics from a specific archived experiment.

        Args:
            phase_name: Name of the phase
            timestamp: Timestamp of the archived experiment

        Returns:
            Metrics dictionary or None if not found
        """
        metrics_path = Path("outputs") / phase_name / "archive" / timestamp / "metrics.json"

        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            return json.load(f)


def migrate_existing_outputs():
    """
    Migrate existing output files to new structure.
    This is a one-time migration script.
    """
    print("=" * 60)
    print("Migrating Existing Outputs to New Structure")
    print("=" * 60)

    # Create base structure
    phases = ["phase1_open", "phase2_slalom", "phase3_maze"]

    for phase in phases:
        base_dir = Path("outputs") / phase
        if not base_dir.exists():
            continue

        print(f"\nProcessing {phase}...")

        # Create latest and archive directories
        latest_dir = base_dir / "latest"
        archive_base = base_dir / "archive"

        latest_dir.mkdir(exist_ok=True)
        archive_base.mkdir(exist_ok=True)

        # Find all files in base directory (not in subdirectories)
        files = [f for f in base_dir.iterdir() if f.is_file()]

        if files:
            # Create archive with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_migrated"
            archive_dir = archive_base / timestamp
            archive_dir.mkdir(exist_ok=True)

            # Move files to archive
            for file in files:
                dest = archive_dir / file.name
                print(f"  Moving {file.name} to archive/{timestamp}/")
                shutil.move(str(file), str(dest))

            # Copy to latest
            for file in archive_dir.iterdir():
                shutil.copy2(str(file), str(latest_dir / file.name))
                print(f"  Copying to latest/: {file.name}")

        print(f"  {phase} migration complete")

    print("\n" + "=" * 60)
    print("Migration Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Run migration
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--migrate":
        migrate_existing_outputs()
    else:
        # Example usage
        print("Output Manager Utility")
        print("\nUsage:")
        print("  python common/output_manager.py --migrate")
        print("    Migrate existing outputs to new structure")
        print("\nIn training scripts:")
        print("  from common.output_manager import OutputManager")
        print("  manager = OutputManager('phase3_maze')")
        print("  manager.save_metrics({...})")
        print("  manager.finalize()")

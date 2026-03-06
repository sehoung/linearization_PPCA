import os
import glob

def cleanup_ppca_files(patterns=None):
    """
    Identifies and prompts the user to delete files matching specific patterns.
    """
    if patterns is None:
        patterns = [
            "./out/figdata/PPCA*",
            "./out/npy/PPCA*",
            "./out/selected_buildings/PPCA*"
        ]

    # Collect all matching file paths
    files_to_remove = []
    for pattern in patterns:
        files_to_remove.extend(glob.glob(pattern))

    if not files_to_remove:
        print("No files found matching the patterns. Nothing to delete.")
        return

    # Show a summary of what will be deleted
    print(f"\n--- Cleanup Preview ---")
    print(f"Found {len(files_to_remove)} files matching PPCA patterns.")
    for f in files_to_remove[:5]:  # Show first 5
        print(f"  [MATCH] {f}")
    if len(files_to_remove) > 5:
        print(f"  ... and {len(files_to_remove) - 5} more.")

    # Interactive confirmation
    confirm = input("\nAre you sure you want to PERMANENTLY delete these files? (y/n): ").lower().strip()

    if confirm == 'y':
        count = 0
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                count += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"Done! Successfully removed {count} files.")
    else:
        print("Cleanup cancelled. No files were deleted.")



def cleanup_trad_files(patterns=None):
    """
    Identifies and prompts the user to delete files matching specific patterns.
    """
    if patterns is None:
        patterns = [
            "./out/figdata/trad*",
            "./out/npy/trad*",
            "./out/selected_buildings/trad*"
        ]

    # Collect all matching file paths
    files_to_remove = []
    for pattern in patterns:
        files_to_remove.extend(glob.glob(pattern))

    if not files_to_remove:
        print("No files found matching the patterns. Nothing to delete.")
        return

    # Show a summary of what will be deleted
    print(f"\n--- Cleanup Preview ---")
    print(f"Found {len(files_to_remove)} files matching PPCA patterns.")
    for f in files_to_remove[:5]:  # Show first 5
        print(f"  [MATCH] {f}")
    if len(files_to_remove) > 5:
        print(f"  ... and {len(files_to_remove) - 5} more.")

    # Interactive confirmation
    confirm = input("\nAre you sure you want to PERMANENTLY delete these files? (y/n): ").lower().strip()

    if confirm == 'y':
        count = 0
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                count += 1
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
        print(f"Done! Successfully removed {count} files.")
    else:
        print("Cleanup cancelled. No files were deleted.")


import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
def reorganize_isles_dataset(source_dir="./data/ISLES-2022",
                             target_dir="./data/ISLES_Handle",
                             num_cases=250):
    """"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return False
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"Source directory: {source_path}")
    print(f"Target directory: {target_path}")
    success_count = 0
    failed_cases = []
    files_found_stats = {"mask": 0, "FLAIR": 0, "dwi": 0, "adc": 0}
    print(f"Starting to process {num_cases} cases...")
    for case_num in tqdm(range(1, num_cases + 1), desc="Processing cases"):
        case_name = f"sub-strokecase{case_num:04d}"
        target_case_dir = target_path / case_name
        target_case_dir.mkdir(exist_ok=True)
        files_copied = []
        missing_files = []
        try:
            mask_source = source_path / "derivatives" / case_name / "ses-0001" / f"{case_name}_ses-0001_msk.nii.gz"
            mask_target = target_case_dir / "mask.nii.gz"
            if mask_source.exists():
                shutil.copy2(mask_source, mask_target)
                files_copied.append("mask.nii.gz")
                files_found_stats["mask"] += 1
            else:
                missing_files.append(f"mask (not found: {mask_source})")
            flair_source = source_path / case_name / "ses-0001" / "anat" / f"{case_name}_ses-0001_FLAIR.nii.gz"
            flair_target = target_case_dir / "FLAIR.nii.gz"
            if flair_source.exists():
                shutil.copy2(flair_source, flair_target)
                files_copied.append("FLAIR.nii.gz")
                files_found_stats["FLAIR"] += 1
            else:
                missing_files.append(f"FLAIR (not found: {flair_source})")
            dwi_source = source_path / case_name / "ses-0001" / "dwi" / f"{case_name}_ses-0001_dwi.nii.gz"
            dwi_target = target_case_dir / "dwi.nii.gz"
            if dwi_source.exists():
                shutil.copy2(dwi_source, dwi_target)
                files_copied.append("dwi.nii.gz")
                files_found_stats["dwi"] += 1
            else:
                missing_files.append(f"DWI (not found: {dwi_source})")
            adc_source = source_path / case_name / "ses-0001" / "dwi" / f"{case_name}_ses-0001_adc.nii.gz"
            adc_target = target_case_dir / "adc.nii.gz"
            if adc_source.exists():
                shutil.copy2(adc_source, adc_target)
                files_copied.append("adc.nii.gz")
                files_found_stats["adc"] += 1
            else:
                missing_files.append(f"ADC (not found: {adc_source})")
            if len(files_copied) == 4:
                success_count += 1
            else:
                if missing_files:
                    failed_cases.append(f"{case_name}: missing {', '.join(missing_files)}")
        except Exception as e:
            print(f"\nError: Error occurred while processing {case_name}: {str(e)}")
            failed_cases.append(f"{case_name}: {str(e)}")
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Successfully processed: {success_count}/{num_cases} cases")
    print(f"\nFile statistics:")
    print(f"  mask files: {files_found_stats['mask']}/{num_cases}")
    print(f"  FLAIR files: {files_found_stats['FLAIR']}/{num_cases}")
    print(f"  dwi files: {files_found_stats['dwi']}/{num_cases}")
    print(f"  adc files: {files_found_stats['adc']}/{num_cases}")
    if failed_cases:
        print(f"\nIncomplete cases ({len(failed_cases)} cases):")
        for case in failed_cases[:10]:
            print(f"  - {case}")
        if len(failed_cases) > 10:
            print(f"  ... and {len(failed_cases) - 10} more cases")
    return True
def check_source_structure(source_dir="/root/autodl-tmp/ISLES-2022"):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return
    print(f"Checking source directory: {source_dir}")
    print("=" * 50)
    print("\nFolders in main directory:")
    main_dirs = []
    for item in sorted(source_path.iterdir()):
        if item.is_dir():
            main_dirs.append(item.name)
            if len(main_dirs) <= 10:
                print(f"  - {item.name}")
    if len(main_dirs) > 10:
        print(f"  ... and {len(main_dirs) - 10} more folders")
    main_cases = [d for d in main_dirs if d.startswith("sub-strokecase")]
    print(f"\nFound {len(main_cases)} sub-strokecase folders in main directory")
    derivatives_path = source_path / "derivatives"
    if derivatives_path.exists():
        derivative_cases = []
        for item in derivatives_path.iterdir():
            if item.is_dir() and item.name.startswith("sub-strokecase"):
                derivative_cases.append(item.name)
        print(f"Found {len(derivative_cases)} sub-strokecase folders in derivatives directory")
    else:
        print("derivatives directory does not exist!")
    print("\nDetailed check of the first case (sub-strokecase0001) structure:")
    case_name = "sub-strokecase0001"
    case_path = source_path / case_name
    if case_path.exists():
        print(f"\n{case_name}/ exists")
        ses_path = case_path / "ses-0001"
        if ses_path.exists():
            print(f"  ses-0001/ exists")
            anat_path = ses_path / "anat"
            if anat_path.exists():
                anat_files = list(anat_path.glob("*.nii.gz"))
                print(f"    anat/: found {len(anat_files)} .nii.gz files")
                for f in anat_files[:5]:
                    print(f"      - {f.name}")
            else:
                print("    anat/: does not exist")
            dwi_path = ses_path / "dwi"
            if dwi_path.exists():
                dwi_files = list(dwi_path.glob("*.nii.gz"))
                print(f"    dwi/: found {len(dwi_files)} .nii.gz files")
                for f in dwi_files[:5]:
                    print(f"      - {f.name}")
            else:
                print("    dwi/: does not exist")
        else:
            print(f"  ses-0001/: does not exist")
            ses_folders = [d for d in case_path.iterdir() if d.is_dir() and d.name.startswith("ses-")]
            if ses_folders:
                print(f"  Found other session folders: {[d.name for d in ses_folders]}")
    else:
        print(f"\nWarning: {case_name} folder not found in main directory!")
    if derivatives_path.exists():
        mask_path = derivatives_path / case_name / "ses-0001"
        if mask_path.exists():
            mask_files = list(mask_path.glob("*.nii.gz"))
            print(f"\nderivatives/{case_name}/ses-0001/: found {len(mask_files)} .nii.gz files")
            for f in mask_files:
                print(f"  - {f.name}")
        else:
            print(f"\nderivatives/{case_name}/ses-0001/: does not exist")
def verify_structure(target_dir="/root/autodl-tmp/ISLES_Handle"):
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Target directory {target_dir} does not exist!")
        return
    print("\nVerifying file structure...")
    print("=" * 50)
    case_dirs = [d for d in target_path.iterdir() if d.is_dir() and d.name.startswith("sub-strokecase")]
    case_dirs.sort()
    total_cases = len(case_dirs)
    complete_cases = 0
    incomplete_cases = []
    for case_dir in case_dirs:
        expected_files = ["mask.nii.gz", "FLAIR.nii.gz", "dwi.nii.gz", "adc.nii.gz"]
        existing_files = [f for f in expected_files if (case_dir / f).exists()]
        if len(existing_files) == 4:
            complete_cases += 1
        else:
            missing = set(expected_files) - set(existing_files)
            incomplete_cases.append(f"{case_dir.name}: missing {missing}")
    print(f"Total cases found: {total_cases}")
    print(f"Complete cases: {complete_cases}/{total_cases}")
    if incomplete_cases:
        print(f"\nIncomplete cases ({len(incomplete_cases)} cases):")
        for case in incomplete_cases[:10]:
            print(f"  - {case}")
        if len(incomplete_cases) > 10:
            print(f"  ... and {len(incomplete_cases) - 10} more cases")
def main():
    parser = argparse.ArgumentParser(description='Reorganize ISLES-2022 dataset')
    parser.add_argument('--source', type=str, default='/root/autodl-tmp/ISLES-2022',
                        help='Source dataset path')
    parser.add_argument('--target', type=str, default='/root/autodl-tmp/ISLES_Handle',
                        help='Target folder path')
    parser.add_argument('--num-cases', type=int, default=250,
                        help='Number of cases')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify target folder structure')
    parser.add_argument('--check-source', action='store_true',
                        help='Check source folder structure')
    args = parser.parse_args()
    if args.check_source:
        check_source_structure(args.source)
    elif args.verify_only:
        verify_structure(args.target)
    else:
        success = reorganize_isles_dataset(args.source, args.target, args.num_cases)
        if success:
            verify_structure(args.target)
if __name__ == "__main__":
    main()

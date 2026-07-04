from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
data_filename = ["adc.nii.gz",
                 "dwi.nii.gz"]
seg_filename = "mask.nii.gz"
def process_train():
    base_dir = "./data"
    image_dir = "ISLES_Handle"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir,
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    out_spacing = [2.0, 2.0, 2.0]
    output_dir = "./data/train_fullres_process/"
    preprocessor.run(output_spacing=out_spacing,
                     output_dir=output_dir,
                     all_labels=[1],
    )
def plan():
    base_dir = "./data"
    image_dir = "ISLES_Handle"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir,
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    analysis_path = "./data/data_analysis_result.txt"
    preprocessor.run_plan(analysis_path)
if __name__ == "__main__":
    plan()
    process_train()

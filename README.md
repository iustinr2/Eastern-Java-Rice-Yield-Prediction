# Eastern-Java-Rice-Yield-Prediction

DRAFT: quick rundown of how to run the code.

First, download the dataset from Hugging Face, and follow all file-reorganizing instructions as large Sentinel-2 files were uploaded on a yearly basis instead of at once (to avoid errors associated with runtimes of multiple days).

Then, download the code, and run in order starting from config -> dataset -> loss -> util. The function of all files will be explained thoroughly in the final README file. After the files in the util folder have been run, I suggest running the remaining files in the order below:

* attention.py
* models_mmst_vit.py
* models_pvt.py
* models_pvt.simclr.py
* main_pretrain_mmst_vit.py
* main_finetune_mmst_vit.py

Within the Evaluation (Chapter 5, Section 5), GradCAM maps are used. To create the GradCAM maps, and GradCAM maps with perturbations, I suggest running the following two files in order:

* create_gradcam_map.py
* gradcam_perturbations.py

To run all files smoothly, ensure that the input directories match the path where your Hugging Face data is stored. All files from Hugging Face are downloadable with the same path (unless specified otherwise), however make sure to replace /vol/home/s3881946 with the root on your device.

Only two python files require manual inputs in the code, build_config_sentinel.py and main_finetune_mmst_vit.py. As build_config_sentinel.py was used for both pretraining and pretraining / finetuning data, it will require you to specify the input and output directory (--input-dir; --output-dir) in the terminal. The Sentinel-2 output size (--output-size) can also be modified, however I recommend using 224 x 224 pixels as this was used in the thesis and strikes a balance between resolution and processed file size. 

Within the fine-tuning file, the split mode in line 29 allows you to select between experiments. The inputs here are either "standard" for random-split, "temporal" for temporal holdout-year, or "spatial" for spatial holdout-regency predictions. Regency embeddings can be turned on using True and off using False (only applicable to random-split and temporal holdout-year experiments) in line 30. For spatial holdout-regency tests, line 30 should remain set to False. Additionally, the output root in line 31 must be modified between runs, otherwise the new experiment will overwrite previous results.  

Two final changes that can be applied in the fine-tuning file are the selected seeds, which can be modified in line 36. Using one, three, or five will make the process finish faster, but also provide less reliable results than using the full ten-seed model. Furthermore, overwrite_existing in line 47 can be set to 0 to ensure that experiments do not repeat fully if they have already been run. For example, if a temporal holdout-year experiment with regency embeddings has already been run, the model will simply output the results for that setting without repeating the fine-tuning process.

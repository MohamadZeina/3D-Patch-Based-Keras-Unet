
# 3D_patch_processing_utils
## About

Useful utilities for working with 3D data in patches. They utilities were designed to train and evaluate deep learning models for 3D segmentation of brain MRI data. Main classes:
 - CategoriseNiftis: this takes a list of niftis, and corresponding SPM segmentations and makes it easy to access raw data and their corresponding segmentation files
 - PatchSequence: this is a keras generator, which inherits the "Sequence" class. This takes a list of file paths, and returns shuffled patches for training.
 - UnetEvaluator: this overrides PatchSequence. It contains various methods for evaluating models. For example, it might take a model and an unsegmented volume, segment it and display the output.

![](raw_image.gif) ![](unet_segmentation.gif)

## Example Usage

    niftis_path = "/path/to/images" # Points to images which have been segmented in SPM 
    
    model = your_keras_model()
    
    niftis = CategoriseNiftis(niftis_path, require_oasis=False, require_string='T1')
    
    generator = PatchSequence(
        [niftis.raw], [niftis.seg_1, niftis.seg_2, niftis.seg_3], 
        batch_size=16, patch_size=128, stride = 8)
        
    history = model.fit_generator(generator, max_queue_size=200, shuffle = False)


## Acknowledgements
Data provided during development by OASIS:
* OASIS-3: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P50AG00561, P30NS09857781, P01AG026276, P01AG003991, R01AG043434, UL1TR000448, R01EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.
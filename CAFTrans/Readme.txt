train:
1. Prepare image segmentation datasets
You should modify the root_path under train or test
root_path has two directories: images and labels
own_data is used to build the Dataset and you can choose whether to pre-process the data or not
custom_transforms is used for data pre-processing

2.Select the model you need to train and the mode or path to save the model
--model_name  eg：FCN，U-net， segcloud， PSP，cloudsegnet 
--model_saved_model eg: tatal model or parameters of the model
--save_path: eg ./result

3.Selection of hyperparameters
n_gpu, lr, batch_size, epoch
location of log saved

This code uses the ploy training strategy, so We need to set max_epoch and max_iterations
You can modify the training strategy yourself


test:
1. Prepare image segmentation datasets
You should modify the root_path under train or test
root_path has two directories: images and labels
own_data is used to build the Dataset and you can choose whether to pre-process the data or not
custom_transforms is used for data pre-processing

2.Select the model to be loaded, mode of loading models, and the model to be evaluated
--load_model eg:'xxx.pth'
--loaded_model_type eg  tatal model or parameters of the model
--evaluation eg:save_image_mask or test_evaluation
If test_evaluation is selected, the output is the Precision, Recall, F-score, Accuracy, and
intersection over union (IoU), of the model
If save_image_mask is selected, the output is mask of the model

When you select save_image_mask, you should modify the --model_name and --save_image_path
The output mask will be saved in the 'save_image_path/model_name/xx'

3.Selection of hyperparameters
n_gpu, lr, batch_size, epoch
location of log saved

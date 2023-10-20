"""
Perform inference for an ensemble of baseline models and save 3D Nifti images of
predicted probability maps averaged across ensemble models (saved to "*pred_prob.nii.gz" files),
binary segmentation maps predicted obtained by thresholding of average predictions and 
removing all connected components smaller than 9 voxels (saved to "pred_seg.nii.gz"),
uncertainty maps for reversed mutual information measure (saved to "uncs_rmi.nii.gz").
"""

import argparse
import os
import glob
import re
import torch
from monai.inferers import sliding_window_inference
from model import UNet3D
from monai.data import write_nifti
import numpy as np
from data_load import remove_connected_components, get_val_dataloader
#from uncertainty import ensemble_uncertainties_classification

parser = argparse.ArgumentParser(description='Get all command line arguments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# save options
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory to store predictions')
# model
#parser.add_argument('--num_models', type=int, default=3,
#                    help='Number of models in ensemble')
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the path to the trained model')
# data
parser.add_argument('--path_data', type=str, required=True,
                    help='Specify the path to the data directory where img/ labels/ (and bm/) directories can be found')
parser.add_argument('--test', action="store_true", default=False, help="whether to use the test set or not. (default to validation set)")
parser.add_argument('--sequences', type=str, nargs='+', required=True,
                    help='input sequences to the model (order is important)')
parser.add_argument('--apply_mask', type=str, default=None, help="Name of the mask to apply")
parser.add_argument('--mode', type=str, required=True,
                    help="segmentation mode (either 'instances' or 'semantic')")
parser.add_argument('--include_ctrl', action="store_true", default=False,
                    help='whether to include the control lesions')
                    
# parallel computation
parser.add_argument('--num_workers', type=int, default=10,
                    help='Number of workers to preprocess images')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.35,
                    help='Probability threshold')

parser.add_argument('--compute_dice', action="store_true", default=False, help="Whether to compute the dice over all the dataset after having predicted it")


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def dice_score(prediction, ground_truth):
    intersection = np.sum(prediction[ground_truth==1])
    dice = (2. * intersection) / (np.sum(prediction) + np.sum(ground_truth))
    return dice

def compute_dice(gt_paths, pred_paths):
    import nibabel as nib
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    avg_dsc = 0
    print("Computing dice on the dataset...")
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        # Load nifti files
        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)

        # Get data from nifti file
        gt = gt_img.get_fdata()
        pred = pred_img.get_fdata()
        
        avg_dsc += dice_score(pred, gt)
    
    avg_dsc /= len(gt_paths)
    print(f"The dice score of the dataset averaged over all the subjects is {avg_dsc}")


def main(args):
  #  os.makedirs(args.path_pred, exist_ok=True)
  #  device = get_default_device()
  #  torch.multiprocessing.set_sharing_strategy('file_system')

  #  '''' Initialise dataloaders '''
  #  val_loader = get_val_dataloader(data_dir=args.path_data,
  #                                  num_workers=args.num_workers,
  #                                  I=args.sequences,
  #                                  mode=args.mode,
  #                                  include_ctrl=bool(args.include_ctrl),
  #                                  test=args.test,
  #                                  apply_mask=args.apply_mask)

  #  ''' Load trained model  '''
  #  in_channels = len(args.sequences)
  #  path_pred = os.path.join(args.path_pred, os.path.basename(os.path.dirname(args.path_model)))
  #  os.makedirs(path_pred, exist_ok=True)

  #  model = UNet3D(in_channels, num_classes=2)
  #  model.load_state_dict(torch.load(args.path_model))
  #  model.to(device)
  #  model.eval()

  #  act = torch.nn.Softmax(dim=1)
  #  th = args.threshold
  #  roi_size = (96, 96, 96)
  #  sw_batch_size = 4

  #  ''' Predictions loop '''
  #  with torch.no_grad():
  #      for count, batch_data in enumerate(val_loader):
  #          inputs = batch_data["image"].to(device)
  #          #foreground_mask = batch_data["brain_mask"].numpy()[0, 0]

  #          # get ensemble predictions
  #          all_outputs = []
  #          outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian')
  #          outputs = act(outputs).cpu().numpy()
  #          outputs = np.squeeze(outputs[0, 1])

  #          # get image metadata
  #          meta_dict = args.sequences[0] + "_meta_dict"
  #          original_affine = batch_data[meta_dict]['original_affine'][0]
  #          affine = batch_data[meta_dict]['affine'][0]
  #          spatial_shape = batch_data[meta_dict]['spatial_shape'][0]
  #          filename_or_obj = batch_data[meta_dict]['filename_or_obj'][0]
  #          filename_or_obj = os.path.basename(filename_or_obj)

  #          # obtain and save prediction probability mask
  #          filename = filename_or_obj[:14] + "_pred_prob.nii.gz"
  #          filepath = os.path.join(path_pred, filename)
  #          write_nifti(outputs, filepath,
  #                      affine=original_affine,
  #                      target_affine=affine,
  #                      output_spatial_shape=spatial_shape)

  #          # obtain and save binary segmentation mask
  #          seg = outputs.copy()
  #          seg[seg >= th] = 1
  #          seg[seg < th] = 0
  #          seg = np.squeeze(seg)
  #          seg = remove_connected_components(seg)

  #          filename = filename_or_obj[:14] + "_seg_binary.nii.gz"
  #          filepath = os.path.join(path_pred, filename)
  #          write_nifti(seg, filepath,
  #                      affine=original_affine,
  #                      target_affine=affine,
  #                      mode='nearest',
  #                      output_spatial_shape=spatial_shape)

     
    path_pred = os.path.join(args.path_pred, os.path.basename(os.path.dirname(args.path_model)))
    if args.compute_dice:
        gt_files = glob.glob(os.path.join(args.path_data, 'test', 'labels', '*mask-classes.nii.gz')) if args.test \
                else glob.glob(os.path.join(args.path_data, 'val', 'labels', '*mask-classes.nii.gz'))
        pred_files = glob.glob(os.path.join(path_pred, '*seg_binary.nii.gz'))
        compute_dice(sorted(gt_files), sorted(pred_files))


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

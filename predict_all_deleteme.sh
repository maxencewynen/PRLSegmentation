#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M3I-FPR-N-4_1-8/best_DSC_M3I-FPR-N-4_1-8_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw MPRAGE_reg-T2starw_T1w
CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M3I-FPR_les-N-4_1-8/best_DSC_M3I-FPR_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw MPRAGE_reg-T2starw_T1w
CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M3I-FPR_wm-N-4_1-8/best_DSC_M3I-FPR_wm-N-4_1-8_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw MPRAGE_reg-T2starw_T1w

#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-F-N-4_1-8/best_DSC_M1I-F-N-4_1-8_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR
CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-F_les-N-4_1-8/best_DSC_M1I-F_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-F_wm-N-5_1-8/best_DSC_M1I-F_wm-N-5_1-8_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR


#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-M-Y-4-4/best_DSC_M1I-M-Y-4-4_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences mag_T2starw
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-M_les-N-4_1-8/best_DSC_M1I-M_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences mag_T2starw
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-M_wm-N-4_31-7/best_DSC_M1I-M_wm-N-4_31-7_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences mag_T2starw



CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-P-N-4_4-8/best_DSC_M1I-P-N-4_4-8_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences phase_T2starw
CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-P_les-N-4_1-8/best_DSC_M1I-P_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences phase_T2starw
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-P_wm-N-4_31-7/best_DSC_M1I-P_wm-N-4_31-7_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences phase_T2starw



#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-R-N-4_31-7/best_DSC_M1I-R-N-4_31-7_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences MPRAGE_reg-T2starw_T1w
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-R_les-N-4_1-8/best_DSC_M1I-R_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences MPRAGE_reg-T2starw_T1w
CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M1I-R_wm-N-4_31-7/best_DSC_M1I-R_wm-N-4_31-7_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences MPRAGE_reg-T2starw_T1w


#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FP-N-4_1-8/best_DSC_M2I-FP-N-4_1-8_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FP_les-N-4_1-8/best_DSC_M2I-FP_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FP_wm-N-4_1-8/best_DSC_M2I-FP_wm-N-4_1-8_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR phase_T2starw


#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FR-N-4_1-8/best_DSC_M2I-FR-N-4_1-8_seed1.pth --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR MPRAGE_reg-T2starw_T1w
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FR_les-N-4_1-8/best_DSC_M2I-FR_les-N-4_1-8_seed1.pth --apply_mask samseg_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR MPRAGE_reg-T2starw_T1w
#CUDA_VISIBLE_DEVICES=0 python inference.py --path_pred ~/data/bxl/predictions/ --path_model /linux/mwynen/models/InstanceSegmentation/M2I-FR_wm-N-4_1-8/best_DSC_M2I-FR_wm-N-4_1-8_seed1.pth --apply_mask wm_dilated_cropped --path_data ~/data/bxl --mode semantic --num_workers 4 --threshold 0.4 --sequences FLAIR MPRAGE_reg-T2starw_T1w



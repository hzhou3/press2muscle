import os
import random
import numpy as np
import torch
from tqdm import tqdm

from dataloader.dataloader import load_data, TestDataset
from models.full_model import TransformerAutoEncoder

from utils.metrics import all_metrics, evaluate_prediction, get_metric_function
from utils.helpers import align_sequence
from utils.parser_util import get_args


cls_metrics = ['f1', 'precision', 'recall']

device = "cuda" if torch.cuda.is_available() else "cpu"


def instance2all(log, instance_log_list):
    skip = 0

    for instance_log in instance_log_list:
        if instance_log.get("muscle_mape", 0) >= np.inf:
            skip = 1
            break 

        for key in instance_log.keys():
            if key not in log:
                log[key] = instance_log[key] 
            else:
                log[key] += instance_log[key]

    return log, skip

def overlapping_test_simplify(args, data, model, num_per_batch=256):
    """
    given a long sequence, split it into batches b for testing
    """

    gt_data, y_gt, filename, uid_raw, act_gt_raw = (data[0], data[1], data[2], data[3], data[4])
    num_frames = gt_data.shape[0]
    gt_data = gt_data.to(device).float()  

    gt_data_splits = []
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = gt_data[:1].repeat(block_seq - 1, 1)
    gt_data_pad = torch.cat((seq_pad, gt_data), dim=0)  

    for i in range(0, num_frames, block_seq): # block_seq
        gt_data_splits.append(gt_data_pad[i: i + block_seq])

    gt_data_splits = torch.stack(gt_data_splits)  

    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    output_samples_y = []
    output_pred_act = []

    for step_index in range(n_steps):
        gt_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        B = gt_per_batch.shape[0]
        uid = uid_raw.repeat(B, 1).to(device)
        
        with torch.no_grad():
            pred_y, pred_act = model(x=gt_per_batch, uid=uid)
        output_samples_y.append(pred_y.cpu().float())
        output_pred_act.append(torch.argmax(pred_act, dim=1).cpu().float() )
    return  output_samples_y, y_gt, output_pred_act, act_gt_raw, filename



def evaluate_prediction_cls(args, metrics, pred, gt, fps, filename, prefix='pressure'):

    # print(pred, gt)
    
    ''' major voting'''
    pred, _ = torch.mode(pred, dim=0, keepdim=True)
    
    gt = gt.unsqueeze(0).expand(pred.shape[0])
    eval_log = {}
    for metric in metrics:
        eval_log[prefix+'_'+metric] = (
            get_metric_function(metric)(
                pred, gt
            ).cpu().numpy()
        )
    torch.cuda.empty_cache()
    return eval_log



def test_process(args=None, log_path=None, split='test', epoch=-1):
    if args is None:
        args = get_args()
    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)

    fps = args.FPS  
    print("Loading dataset...")
    inputs, outputs, files = load_data(
        args.DATASET_PATH,
        split,
        protocol=args.PROTOCOL,
        input_motion_length=args.INPUT_MOTION_LENGTH,
        input_dim=args.INPUT_DIM,
        output_dim=args.OUTPUT_DIM,
    )

    dataset = TestDataset(inputs, outputs, files, with_context=True)

    log = {} 
    
    in_dim = args.INPUT_DIM
    out_dim = args.OUTPUT_DIM 

    model_cfg = args.MODEL_CFG
    block_size = args.INPUT_MOTION_LENGTH + 1
    model = TransformerAutoEncoder(in_dim=in_dim, out_dim=out_dim, n_layers=model_cfg.n_layers, hid_dim=model_cfg.hid_dim, heads=model_cfg.heads,
                             dropout=model_cfg.dropout, embed_type=model_cfg.embed_type, block_size=block_size)
    
    model = model.to(device)
    model.eval()

    output_dir = args.SAVE_DIR
    model_file = os.path.join(output_dir, 'best.pth.tar')
    if os.path.exists(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        print(f"{model_file} not exist!!!")
        # return

    n_testframe = args.NUM_PER_BATCH
    file_str = ""
    skip = 0
    for sample_index in tqdm(range(len(dataset))):
        y_pred, y_gt, _, _, filename = \
            overlapping_test_simplify(args, dataset[sample_index], model, n_testframe)
        y_pred = torch.cat(y_pred, dim=0)  # (N, 8) (frames, out_dim)
        y_pred, y_gt = align_sequence(y_pred, y_gt, args)

        if y_pred.shape[0] < args.INPUT_MOTION_LENGTH:
            skip += 1
            continue
        
        log_y = evaluate_prediction(
            args, all_metrics, y_pred, y_gt, fps, filename, prefix='muscle')
        
        file_str += filename + f",{log_y['muscle_mape']},{log_y['muscle_rmse']},{log_y['muscle_rho']}\n"

        log, sub_skip = instance2all(log, [log_y])
        skip += sub_skip
        
    # Print the value for all the metrics
    print("Metrics for the predictions")
    result_str = "\n".join([f"{metric}: {log[metric] / (len(dataset)-skip)}" for metric in log.keys()])
    print(result_str)

    if log_path is not None:
        with open(log_path, 'a') as f:
            f.write(result_str)
        print(f"Evalution results save to {log_path}\n")

        with open(log_path.replace('.log', "_details.log"), 'a') as f:
            file_str += '==================\n'
            f.write(file_str)
    

if __name__ == "__main__":
    test_process()

import pandas as pd
import numpy as np
import random, argparse, glob
from tqdm import tqdm
import os



def from_long_file_to_short_file(f):
    return os.path.basename(f).split(".")[0].split('_')[0].lower()


def remove_ignore(files, ignore):

    if ignore is None:
        return files
    return [f for f in files if from_long_file_to_short_file(f) not in ignore]

def create_ignore(users: list, visits: list):
    return [f"u{int(user)}v{int(visit)}" for user in users for visit in visits]

def ignore_files(ignore):

    if ignore:
        dpath='./dataset/ignore.txt'
    else:
        dpath = ""
        
    if not os.path.exists(dpath):
        print("==> ignore file not found")
        return None
    
    with open(dpath, "r") as f:
        lines = f.readlines()
    ignores = [line.strip().lower() for line in lines]
    print(f"==> ignoring {len(ignores)} files")
    return ignores



def shift_pressure(df, max_shift=5, shift_prob=0.5):
    shifted_df = df.copy()
    gt_data = df.iloc[:, 9:].values  # Extract ground truth (gt)
    
    for i in range(gt_data.shape[1]):  # Iterate over 8 gt channels
        if np.random.rand() < shift_prob:  # Only shift some channels
            shift = np.random.randint(-max_shift, max_shift + 1)  # Random shift per channel
            gt_data[:, i] = np.roll(gt_data[:, i], shift)

            # Fill missing values (boundary effect)
            if shift > 0:
                gt_data[:shift, i] = gt_data[shift, i]  # Fill first `shift` rows
            elif shift < 0:
                gt_data[shift:, i] = gt_data[shift - 1, i]  # Fill last `shift` rows

    shifted_df.iloc[:, 9:] = gt_data  # Replace with modified gt
    return shifted_df


def scale_pressure(
    df,
    min_scale=0.8,
    max_scale=1.2,
    scale_prob=0.5,
    q_range=(50, 90),          # dynamic threshold percentile range
    min_active_frac=0.01,      # skip channels with too little activity
    eps=1e-8,
):
    scaled_df = df.copy()
    gt = df.iloc[:, 9:].values.astype(np.float32)  # (T, C)

    for c in range(gt.shape[1]):
        if np.random.rand() >= scale_prob:
            continue

        x = gt[:, c]

        # focus threshold on "active" region so zeros don't dominate
        active = x > 0
        if active.mean() < min_active_frac:
            continue

        # dynamic threshold for this channel in this sample
        q = np.random.uniform(*q_range)
        thr = np.percentile(x[active], q) + eps

        # soft weight: 0 below thr, ramps toward 1 above thr
        w = np.clip(x / thr, 0.0, 1.0)

        s = np.random.uniform(min_scale, max_scale)

        # soft scaling: below thr almost unchanged; above thr scaled more
        gt[:, c] *= (1.0 + w * (s - 1.0))

    scaled_df.iloc[:, 9:] = gt
    return scaled_df

# def scale_pressure(df, min_scale=0.8, max_scale=1.2, scale_prob=0.5):
#     scaled_df = df.copy()
#     gt_data = df.iloc[:, 9:].values  # Extract ground truth (gt)
    
#     for i in range(gt_data.shape[1]):  # Iterate over 8 gt channels
#         if np.random.rand() < scale_prob:  # Only scale some channels
#             scale_factor = np.random.uniform(min_scale, max_scale)
#             gt_data[:, i] *= scale_factor  # Apply scaling

#     scaled_df.iloc[:, 9:] = gt_data  # Replace with modified gt
#     return scaled_df
    
def apply_random_augmentations(fname):
    df = pd.read_csv(fname)
    """ Apply a random combination of augmentations to the dataset. """
    aug_functions = [
        (shift_pressure, "shift"), 
        (scale_pressure, "scalepressure"),

    ]
    np.random.shuffle(aug_functions)  # Randomize application order

    for (aug, fc_id) in aug_functions:  # Apply augmentations
        df = aug(df)
        df.to_csv(fname.replace('.csv', f'_{args.id}_{fc_id}.csv'), index=False)
        

def main(args):
    random.seed(args.seed)
    files = glob.glob(args.root + f"/{args.folder}/u{args.user}*.csv")

    if args.remove: # remove augmented files
        rm_files = [f for f in files if "augment" in f]
        for f in rm_files:
            os.remove(f)
        
    files = [f for f in files if "augment" not in f]

    files = remove_ignore(files, ignore_files(args.ignore))
    
    for f in tqdm(files):
        apply_random_augmentations(f)
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./dataset")
    parser.add_argument("-f", "--folder", type=str, default="processed") 
    parser.add_argument("-u", "--user", type=str, default="") 
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("-i", "--ignore", action="store_true")
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--id", type=str, default="augment1")
    
    args = parser.parse_args()

    main(args)
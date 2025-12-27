import glob
import os
import random
import argparse
from .dataset_details import KEEP_LABELS, ACTIVITY_DETAILS, ACTIVITY_LABELS
import random
from collections import defaultdict

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


def from_long_file_to_short_file(f):
    return os.path.basename(f).split(".")[0].split('_')[0].lower()


def remove_ignore(files, ignore):

    if ignore is None:
        return files
    return [f for f in files if from_long_file_to_short_file(f) not in ignore]


def clean_files(files):
    
    def get_label(fname):
        return (ACTIVITY_DETAILS[fname] in KEEP_LABELS)
    
    return [f for f in files if get_label(from_long_file_to_short_file(f))] 



def save_files(train_files, test_files, args, val_ratio=0.1, seed=0):
    """
    Save train / val / test protocol files.
    Train/val split is done by user (e.g., u2v16.csv -> user u2).
    """

    if len(train_files) == 0 or len(test_files) == 0:
        print("==> train or test is empty to save")
        return

    random.seed(seed)

    # ----------------------------
    # group train files by user
    # ----------------------------
    user_to_files = defaultdict(list)
    for f in train_files:
        fname = os.path.basename(f)
        user = fname.split("v")[0]   # "u2v16.csv" -> "u2"
        user_to_files[user].append(f)

    users = list(user_to_files.keys())
    random.shuffle(users)

    n_val_users = max(1, int(len(users) * val_ratio))
    val_users = set(users[:n_val_users])
    train_users = set(users[n_val_users:])

    train_split, val_split = [], []

    for u in train_users:
        train_split.extend(user_to_files[u])

    for u in val_users:
        val_split.extend(user_to_files[u])

    random.shuffle(train_split)
    random.shuffle(val_split)

    # ----------------------------
    # save files
    # ----------------------------
    protocol_dir = os.path.join(args.root, "protocols")
    os.makedirs(protocol_dir, exist_ok=True)

    with open(os.path.join(protocol_dir, f"{args.protocol}_train.txt"), "w") as f:
        for x in train_split:
            f.write(x + "\n")

    with open(os.path.join(protocol_dir, f"{args.protocol}_val.txt"), "w") as f:
        for x in val_split:
            f.write(x + "\n")

    with open(os.path.join(protocol_dir, f"{args.protocol}_test.txt"), "w") as f:
        for x in test_files:
            f.write(x + "\n")

    print(
        f"Saved protocol '{args.protocol}': "
        f"{len(train_users)} train users, "
        f"{len(val_users)} val users, "
        f"{len(test_files)} test files"
    )

   
def get_user(file_name):
    return int(os.path.basename(file_name).split(".")[0].split('v')[0].replace('u', ""))


def get_visit(file_name):
    return int(os.path.basename(file_name).split(".")[0].split('v')[1].replace('v', ""))





##### get random mode

def get_random_mode(files, args):
    random.shuffle(files)
    train_files = files[:int(len(files) * args.ratio)]
    test_files = files[int(len(files) * args.ratio):]

    return train_files, test_files


##### get user mode
def get_user_all(files, args):

    for trg_user in range(1, args.N):
        args.protocol = f"user-{trg_user}"
        train_files, test_files = get_user_mode(files, args)
        if len(train_files) == 0 or len(test_files) == 0:
            break
        print(f"==> making user-{trg_user} split with {len(train_files)} train and {len(test_files)} test files")    
        save_files(train_files, test_files, args)
    

def get_user_mode(files, args):
    train_files = []
    test_files = []
    trg_user = [int(i) for i in args.protocol.split("-")[1:]]
    # trg_user = [args.protocol.split("-")[1].split("_")[0]]

    trg_aug_files = [f for f in files if (get_user(f) in trg_user) and ('_augmented' in f)]

    files = list(set(files) - set(trg_aug_files))

    for f in files:
        # if "_augmented" in f:
        #     continue

        user = get_user(f)

        if user in trg_user and "_augmented" not in f:
            test_files.append(f)
        else:

            train_files.append(f)

    print(f"==> train files: {len(train_files)}, test files: {len(test_files)}")

    return train_files, test_files








def make_split(args):
    
    random.seed(args.seed)
    files = glob.glob(args.root + f"/{args.folder}/u*.csv")
    files = remove_ignore(files, ignore_files(args.ignore))

    if args.protocol == 'random':
        train_files, test_files = get_random_mode(files, args)
        print(f"==> making {args.protocol} split")    
        save_files(train_files, test_files, args)

    elif "user-" in args.protocol: 
        train_files, test_files = get_user_mode(files, args)
        print(f"==> making {args.protocol} split")    
        save_files(train_files, test_files, args)

    elif args.protocol == 'user':
        get_user_all(files, args)

    else:
        print("==> protocol not found")
        return

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./dataset")
    parser.add_argument("-f", "--folder", type=str, default="processed") 
    parser.add_argument("-p", "--protocol", type=str, default="random")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--N", type=int, default=31)
    parser.add_argument("-r", "--ratio", type=float, default=0.8)
    parser.add_argument("-i", "--ignore", action="store_true")
    parser.add_argument("--num_workers", type=int, default=32)
    
    args = parser.parse_args()

    make_split(args)

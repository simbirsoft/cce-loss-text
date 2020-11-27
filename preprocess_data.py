"""
This script preprocesses data for the experiment:
1. Relabels classes
2. Resamples classes into desired class proportions
3. Prepares train/test split
"""

import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split


RND_SEED = 123


def relabel_sentiments(df):
    df["Sentiment"] = df["Sentiment"].map({
        0: 0,
        1: 0,
        2: 1,
        3: 2,
        4: 2,
    })
    return df

def resample_classes(df, ratio_0, ratio_p1):
    df_neg = df[df["Sentiment"] == 0]
    num_neg = len(df_neg)
    
    df_neu = df[df["Sentiment"] == 1].sample(n=int(num_neg*ratio_0), random_state=RND_SEED)
    df_pos = df[df["Sentiment"] == 2].sample(n=int(num_neg*ratio_p1), random_state=RND_SEED)

    df = pd.concat([df_neg, df_neu, df_pos]).sample(frac=1, random_state=RND_SEED)

    print(f"Ratios: [1, {ratio_0}, {ratio_p1}]; Counts: [{num_neg}, {len(df_neu)}, {len(df_pos)}]")
    return df

def split(df, val_frac, test_frac):
    train, test = train_test_split(df, test_size=test_frac, 
        shuffle=True, random_state=RND_SEED)
    train, val = train_test_split(train, test_size=val_frac*(1-test_frac), 
        shuffle=True, random_state=RND_SEED)
    
    train = pd.DataFrame(train, columns=df.columns)
    val = pd.DataFrame(val, columns=df.columns)
    test = pd.DataFrame(test, columns=df.columns)

    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="File to process")
    parser.add_argument("out_folder", help="Output directory")
    parser.add_argument("--val_frac", help="Fraction of data to use for validation set", type=float, default=0.1)
    parser.add_argument("--test_frac", help="Fraction of data to use for test set", type=float, default=0.2)
    parser.add_argument("--mid_class_ratio", help="Ratio of a middle-sized class", type=float, default=0.5)
    parser.add_argument("--low_class_ratio", help="Ratio of a low-sized class", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.file_name, sep="\t")
    df = df[["Phrase", "Sentiment"]]
    df = relabel_sentiments(df)

    ratios = [
        (1, 1),
        (args.mid_class_ratio, args.mid_class_ratio),
        (args.mid_class_ratio, args.low_class_ratio),
        (args.low_class_ratio, args.low_class_ratio),
    ]

    for (ratio_0, ratio_p1) in ratios:
        df = resample_classes(df, ratio_0, ratio_p1)
        train, val, test = split(df, args.val_frac, args.test_frac)
        train.to_csv(os.path.join(args.out_folder, f"train_{ratio_0:.2f}_{ratio_p1:.2f}.csv"), index=False)
        val.to_csv(os.path.join(args.out_folder, f"val_{ratio_0:.2f}_{ratio_p1:.2f}.csv"), index=False)
        test.to_csv(os.path.join(args.out_folder, f"test_{ratio_0:.2f}_{ratio_p1:.2f}.csv"), index=False)

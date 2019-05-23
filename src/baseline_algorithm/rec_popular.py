from pathlib import Path

import click
import pandas as pd

# from . import functions as f
import src.baseline_algorithm.functions as f

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default='/home/jayg996/recsyschallenge/data', help='Directory for the CSV files')
def main(data_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('toy_train.csv')
    test_csv = data_directory.joinpath('toy_test.csv')
    subm_csv = data_directory.joinpath('toy_submission_popular.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    print("Get popular items...")
    df_popular = f.get_popularity(df_train)

    print("Identify target rows...")
    df_target = f.get_submission_target(df_test)

    print("Get recommendations...")
    df_expl = f.explode(df_target, "impressions")
    df_out = f.calc_recommendation(df_expl, df_popular)

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
